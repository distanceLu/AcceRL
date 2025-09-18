import os
import sys
from functools import partial
from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import math
import cv2

import logging
mainlogger = logging.getLogger('mainlogger')
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import pdb
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from copy import deepcopy
import time
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import av
import glob
import matplotlib.cm as cm

from utils.general_utils import instantiate_from_config
from lvdm.ema import LitEma
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr
from lvdm.basics import disabled_train
from lvdm.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)

from lvdm.data.utils import gen_batch_ray_parellel, intrinsic_transform_batch, get_transformation_matrix_from_quat
from lvdm.data.domain_table import DomainTable
from lvdm.data.traj_vis_statistics import ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight, EndEffectorPts, Gripper2EEFCvt


'''
Sampler for the logit-normal distribution.
'''
def logit_normal_sampler(m, s=1, beta_m=100, sample_num=1000000):
    y_samples = torch.randn(sample_num) * s + m
    x_samples = beta_m * (torch.exp(y_samples) / (1 + torch.exp(y_samples)))
    return x_samples
'''
the \mu(t) function
'''
def mu_t(t,a=5, mu_max=4):
    t = t.to('cpu')
    return 2 * mu_max * t ** a - mu_max
'''
get beta_s
'''  
def get_beta_s(t, a=5,beta_m=100):
    mu = mu_t(t, a=a)
    beta_s = logit_normal_sampler(m=mu, beta_m=beta_m,sample_num=t.shape[0])
    return beta_s



class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor=None,
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 rescale_betas_zero_snr=False,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = getattr(unet_config.params, "temporal_length")
        self.image_size = image_size  # try conv?
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)

        self.use_ema = use_ema
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        ## for reschedule
        self.given_betas = given_betas
        self.beta_schedule = beta_schedule
        self.timesteps = timesteps
        self.cosine_s = cosine_s

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))

        if self.parameterization != 'v':
            self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        else:
            self.register_buffer('sqrt_recip_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        mainlogger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True, last_only=0):
        if last_only > 0:
            target = target[:,:,-last_only:]
            pred = pred[:,:,-last_only:]
        
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()

        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')

        elif self.loss_type == 'huber':
            if mean:
                loss = torch.nn.functional.smooth_l1_loss(target, pred, beta=0.1)
            else:
                loss = torch.nn.functional.smooth_l1_loss(target, pred, reduction='none', beta=0.1)
                
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, random_uncond=False)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, random_uncond=False)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="caption",
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 uncond_prob=0.2,
                 uncond_type="empty_seq",
                 scale_factor=1.0,
                 scale_by_std=False,
                 encoder_type="2d",
                 only_model=False,
                 noise_strength=0,
                 use_dynamic_rescale=False,
                 base_scale=0.7,
                 turning_step=400,
                 loop_video=False,
                 fps_condition_type='fs',
                 perframe_ae=False,
                 # added
                 logdir=None,
                 rand_cond_frame=False,
                 ae_batch_size=1,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, 'crossattn')
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.noise_strength = noise_strength
        self.use_dynamic_rescale = use_dynamic_rescale
        self.loop_video = loop_video
        self.fps_condition_type = fps_condition_type
        self.perframe_ae = perframe_ae
        self.ae_batch_size = ae_batch_size

        self.logdir = logdir
        self.rand_cond_frame = rand_cond_frame

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_timesteps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config        
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert(encoder_type in ["2d", "3d"])
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert(uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True
                
    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and \
                not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            mainlogger.info("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            mainlogger.info(f"setting self.scale_factor to {self.scale_factor}")
            mainlogger.info("### USING STD-RESCALING ###")
            mainlogger.info(f"std={z.flatten().std()}")

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
    
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None,mode=False):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise) # posterior add noise
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        elif isinstance(encoder_posterior,AutoencoderKLOutput):
            if mode:
                z = encoder_posterior.latent_dist.mode()
            else:
                z = encoder_posterior.latent_dist.sample()
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
   
    @torch.no_grad()
    def encode_first_stage(self, x, mode=False):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
        
        ## consume more GPU memory but faster
        if not self.perframe_ae:
            encoder_posterior = self.first_stage_model.encode(x)
            results = self.get_first_stage_encoding(encoder_posterior,mode=mode).detach()

        elif self.ae_batch_size>1:
            results = []
            n_batch = int(math.ceil(x.shape[0]/float(self.ae_batch_size)))
            for index in range(0, n_batch):
                sidx = index*self.ae_batch_size
                eidx = min(x.shape[0], (index+1)*self.ae_batch_size)
                frame_batch = self.first_stage_model.encode(x[sidx:eidx,:,:,:])
                frame_result = self.get_first_stage_encoding(frame_batch,mode=mode).detach()
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        else:  ## consume less GPU memory but slower
            results = []
            for index in range(x.shape[0]):
                frame_batch = self.first_stage_model.encode(x[index:index+1,:,:,:])
                frame_result = self.get_first_stage_encoding(frame_batch,mode=mode).detach()
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        
        return results
    
    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
            
        if not self.perframe_ae:    
            z = 1. / self.scale_factor * z
            results = self.first_stage_model.decode(z, **kwargs)
            if isinstance(results, DecoderOutput):
                results = results.sample
        else:
            results = []
            for index in range(z.shape[0]):
                frame_z = 1. / self.scale_factor * z[index:index+1,:,:,:]
                frame_result = self.first_stage_model.decode(frame_z, **kwargs)
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)
    
    @torch.no_grad()
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False):
        ## video shape: b, c, t, h, w
        x = super().get_input(batch, self.first_stage_key)

        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)
                
        ## get caption condition
        cond = batch[self.cond_stage_key]
        if random_uncond and self.uncond_type == 'empty_seq':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(cond_emb[i])
        
        out = [z, cond_emb]
        ## optional output: self-reconst or caption
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond)

        return out

    def forward(self, x, c, t=None, motion_mask=None, **kwargs):
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c = self.get_batch_input(batch, random_uncond=random_uncond)
        loss, loss_dict = self(x, c, **kwargs)

        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, **kwargs):
        
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        if self.logvar.device is not self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        # logvar_t = self.logvar[t.item()].to(self.device) # device conflict when ddp shared
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict  

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        '''
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        '''
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss
    
    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W
        
        if denoise_row.dim() == 5:
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=200, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, **kwargs):
        """ log images for LatentDiffusion """
        ##### control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 2
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        use_ddim = ddim_steps is not None
        log = dict()
        z, c, xrec, xc = self.get_batch_input(batch, random_uncond=False,
                                                return_first_stage_outputs=True,
                                                return_original_cond=True)

        N = xrec.shape[0]
        log["reconst"] = xrec
        log["condition"] = xc
        

        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_cat, c_emb = c["c_concat"][0], c["c_crossattn"][0]
                    log["condition_cat"] = c_cat
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc = torch.zeros_like(c_emb)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False, score_corrector=None, corrector_kwargs=None, **kwargs):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_x0=False, \
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, **kwargs):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, \
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, **kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, \
                      timesteps=None, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None, **kwargs):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]        
        # sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, \
               verbose=True, timesteps=None, mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps,
                                  mask=mask, x0=x0, **kwargs)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, causal=False, chunk=4,**kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, causal=causal,chunk=chunk,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True,causal=causal,chunk=chunk,**kwargs)

        return samples, intermediates

    def configure_schedulers(self, optimizer):
        assert 'target' in self.scheduler_config
        scheduler_name = self.scheduler_config.target.split('.')[-1]
        interval = self.scheduler_config.interval
        frequency = self.scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                            'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                            'interval': interval,
                            'frequency': frequency
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                            'scheduler': CosineAnnealingLR(optimizer, T_max=decay_steps, last_epoch=last_step),
                            'interval': interval,
                            'frequency': frequency
            }
        else:
            raise NotImplementedError
        return lr_scheduler


class ACWMLatentDiffusion(LatentDiffusion):
    def __init__(self, img_cond_stage_config, image_proj_stage_config, freeze_embedder=True, unet_trainable_list=None,
                image_proj_model_trainable=True,  traj_prompt_proj_model_trainable=True,
                chunk=16, n_view=1, use_raymap_dir=False, use_raymap_origin=False, 
                use_cat_mask=False, sparse_memory=False, ddim_num_chunk=5,
                views_name_wo_imgcat=[], 
                *args, **kwargs
            ):
        super().__init__(*args, **kwargs)
        assert len(self.first_stage_key) > 1, "self.first_stage_key must be more than 2 for Action Latent"

        self.image_proj_model_trainable = image_proj_model_trainable
        self._init_embedder(img_cond_stage_config, freeze_embedder)
        self._init_img_ctx_projector(image_proj_stage_config, image_proj_model_trainable)
        self.unet_trainable_list = unet_trainable_list

        self.chunk = chunk
        self.n_view = n_view

        self.use_cat_mask = use_cat_mask

        self.use_raymap_dir = use_raymap_dir
        self.use_raymap_origin = use_raymap_origin

        self.count = 0

        self.traj_prompt_proj_model_trainable = traj_prompt_proj_model_trainable
        self.views_name_wo_imgcat = views_name_wo_imgcat

        assert(self.model.conditioning_key == 'hybrid')

        self.sparse_memory = sparse_memory
        self.ddim_num_chunk = ddim_num_chunk


    def _init_img_ctx_projector(self, config, trainable):
        self.image_proj_model = instantiate_from_config(config)
        if not trainable:
            self.image_proj_model.eval()
            self.image_proj_model.train = disabled_train
            for param in self.image_proj_model.parameters():
                param.requires_grad = False

    def _init_embedder(self, config, freeze=True):
        self.embedder = instantiate_from_config(config)
        if freeze:
            self.embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False


    def get_batch_input(self, batch, random_uncond, 
                        return_first_stage_outputs=False, 
                        return_original_cond=False, 
                        return_fs=False, 
                        return_did=False,
                        return_traj=True,
                        return_img_emb=False,
                        pre_z=None,
                        pre_img_emb=None,
                        **kwargs):

        ## x: b c v t h w
        x = super().get_input(batch, self.first_stage_key[0])
        
        if len(x.shape) < 6 and self.n_view==1:
            x = x.unsqueeze(dim=2)

        b, c, v, t, h, w = x.shape
        x  = rearrange(x, 'b c v t h w -> (b v) c t h w')

        if pre_z is not None:
            ### used in inference stage only, replace z with pre_z
            z = pre_z
        else:
            ## encode video frames x to z via a 2D encoder
            z = self.encode_first_stage(x)

        _, _, _, latent_h, latent_w = z.shape


        # get trajectories
        traj = super().get_input(batch, self.first_stage_key[1])

        if len(traj.shape) < 6 and self.n_view==1:
            traj = traj.unsqueeze(dim=2)

        traj_ori  = rearrange(traj, 'b c v t h w -> (b v) c t h w')
        traj = self.encode_first_stage(traj_ori)


        # prepare conditions
        cond_input = batch[self.cond_stage_key]
        cond = {}
        if random_uncond:
            random_num = torch.rand(b, device=x.device)
        else:
            random_num = torch.ones(b, device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference

        prompt_mask = 1 - rearrange((random_num < 2 * self.uncond_prob).float(), "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")
        cat_mask = 1 - (rearrange(
            (random_num < 0.5*self.uncond_prob).float() + 
            ((random_num >= 1 * self.uncond_prob).float()*(random_num < 1.5 * self.uncond_prob).float())  + 
            ((random_num >= 2.5 * self.uncond_prob).float()*(random_num < 3.5 * self.uncond_prob).float()), "n -> n 1 1 1 1")>0).float()
        cat_traj_mask = 1 - (rearrange(
            ((random_num >= 0.5 * self.uncond_prob).float()*(random_num < 1.5 * self.uncond_prob).float()), "n -> n 1 1 1 1")>0).float()

        ## get conditioning frame
        cond_frame_index = t + batch['cond_id']

        if pre_img_emb is None:
            img = []
            x = rearrange(x, '(b v) c t h w -> b c v t h w', b=b)
            for idx_b, cid in enumerate(cond_frame_index):
                if self.rand_cond_frame and cid<self.model.diffusion_model.temporal_length-self.chunk-1:
                    cid_ = random.randint(cid, self.model.diffusion_model.temporal_length-self.chunk-1)
                else:
                    cid_ = cid
                img.append(x[idx_b,:,:,cid_])
            img_ = rearrange(torch.stack(img, dim=0), 'b c v h w -> (b v) c h w')
            x = rearrange(x ,'b c v t h w -> (b v) c t h w')

            input_mask = input_mask.unsqueeze(1).repeat(1,v,1,1,1)
            input_mask = rearrange(input_mask, 'b v c h w -> (b v) c h w')

            img = input_mask * img_

            ## img: (b v) c h w
            img_emb = self.embedder(img)
            img_emb = self.image_proj_model(img_emb)

        else:
            img_emb = pre_img_emb

        cond_input = prompt_mask * cond_input

        cond_emb = self.cond_stage_model(cond_input.to(dtype=x.dtype, device=self.device))
        
        cat_mask = cat_mask.unsqueeze(1).repeat(1,v,1,t,1,1)
        cat_mask = rearrange(cat_mask, 'b v c t h w -> (b v) c t h w')
        cat_mask[:,:,:-self.chunk] = 1.
        cat_mask = cat_mask.to(device=z.device, dtype=z.dtype)

        cat_traj_mask = cat_traj_mask.unsqueeze(1).repeat(1,v,1,t,1,1)
        cat_traj_mask = rearrange(cat_traj_mask, 'b v c t h w -> (b v) c t h w')
        cat_traj_mask[:,:,:-self.chunk] = 0.

        if self.model.conditioning_key == 'hybrid':
            ## simply repeat the cond_frame to match the seq_len of z
            img_cat_cond = z.clone()
            img_cat_cond[:,:,-self.chunk:] = img_cat_cond[:,:,-(self.chunk+1):-self.chunk].repeat(1,1,self.chunk,1,1) # next 8 is the repeat of current
            if self.use_cat_mask:
                img_cat_cond = img_cat_cond*cat_mask    
                cat_mask = cat_mask*torch.ones_like(img_cat_cond[:,:1])
                traj = traj*cat_traj_mask    
                cat_traj_mask = cat_traj_mask*torch.ones_like(traj[:,:1])
            cond["c_concat"] = [img_cat_cond, ]
            if self.use_cat_mask:
                cond['c_concat'].append(cat_mask)
            cond['c_concat'].append(traj)

        else:
            raise NotImplementedError

        cond_emb = rearrange(cond_emb.unsqueeze(dim=1).repeat(1, v, 1, 1), "b v c t -> (b v) c t")
        
        cond["c_crossattn"] = [torch.cat([img_emb, cond_emb], dim=1)] ## concat in the seq_len dim

        if self.use_raymap_dir or self.use_raymap_origin:
            with torch.no_grad():
                intrinsic = super().get_input(batch, 'intrinsic')
                extrinsic = super().get_input(batch, 'extrinsic')

                if len(intrinsic.shape)<4:
                    intrinsic = intrinsic.unsqueeze(dim=1)
                if len(extrinsic.shape)<5:
                    extrinsic = extrinsic.unsqueeze(dim=1)

                intrinsic = intrinsic.unsqueeze(2).repeat(1,1,t,1,1)
                intrinsic = rearrange(intrinsic, 'b v t h w -> (b v t) h w')
                extrinsic = rearrange(extrinsic, 'b v t h w -> (b v t) h w')
                intrinsic = intrinsic_transform_batch(intrinsic, (h, w), (latent_h, latent_w), transform_mode='resize')
                _, batch_raymap_o, batch_raymap_d = gen_batch_ray_parellel(intrinsic, extrinsic, latent_w, latent_h) # shape (b v) latent_h, latent_w, 3

                if self.use_raymap_dir:
                    batch_raymap_d = rearrange(batch_raymap_d, '(bv t) h w c -> bv c t h w', t=t)
                    batch_raymap_d = batch_raymap_d.to(device=z.device)

                if self.use_raymap_origin:
                    batch_raymap_o = rearrange(batch_raymap_o, '(bv t) h w c -> bv c t h w', t=t)
                    batch_raymap_o = batch_raymap_o.to(device=z.device)

            if self.use_raymap_dir:
                cond['c_concat'].append(batch_raymap_d)

            if self.use_raymap_origin:
                cond['c_concat'].append(batch_raymap_o)

        out = [z, cond]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        
        if return_did:
            did = super().get_input(batch, 'domain_id')
            out.append(did)

        if return_traj:
            out.append(traj_ori)

        if return_img_emb:
            out.append(img_emb)

        return out

    def teacher_forcing_aug(self,x):
        """
        """
        b,c,f,h,w = x.shape
        aug_t = torch.rand((b,f), device=self.device)*3.14 - (3.14/2)
        aug_t = aug_t.clamp(0,(3.14/2))
        div = random.randint(3,20)
        aug_t = ((-torch.cos(aug_t) + 1)*(self.num_timesteps/div)).long()
        aug_t,_ = torch.sort(aug_t,dim=1) # b f
        aug_t = rearrange(aug_t,'b f -> (b f)') # b f
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        noise = torch.randn_like(x,device=self.device)
        x_aug = self.q_sample(x_start=x, t=aug_t, noise=noise) # (b f) c h w
        x_aug = rearrange(x_aug,'(b f) c h w -> b c f h w',b=b)
        return x_aug


    def shared_step(self, batch, random_uncond, **kwargs):
        batch_inputs = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True, return_did=True, return_traj=True)
        x, c, fs, did, traj = batch_inputs[:5]
        kwargs.update({"fs": fs.long()})
        kwargs.update({"domain_id": did.long()})
        loss, loss_dict = self(x, c, traj, **kwargs)
        return loss, loss_dict
    

    def forward(self, x, c, traj, t=None, traj_aug_mask=None, **kwargs):
        
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x.shape[0]//self.n_view,), device=self.device).long()
            t = t.unsqueeze(1).repeat(1,self.n_view)
            t = rearrange(t, 'b v -> (b v)')

        if self.training:
            x[:,:,:-self.chunk] = self.teacher_forcing_aug(x[:,:,:-self.chunk])
            c['c_concat'][0] = torch.cat([x[:,:,:-self.chunk], x[:,:,-(self.chunk+1):-self.chunk].repeat(1,1,self.chunk,1,1)],dim=2)
            if self.use_cat_mask:
                c['c_concat'][0] = c['c_concat'][0]*c['c_concat'][1]

        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        
        return self.p_losses(x, c, t, traj_aug_mask=traj_aug_mask, **kwargs)


    def p_losses(self, x_start, cond, t, noise=None, traj_aug_mask=None, **kwargs):

        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy[:,:,:-self.chunk] = x_start[:,:,:-self.chunk]

        model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple_map = self.get_loss(model_output, target, mean=False, last_only=self.chunk)
        if traj_aug_mask is not None:
            traj_aug_mask = 1.0-traj_aug_mask[:,:,-self.chunk:]
            loss_simple_map = ((loss_simple_map * traj_aug_mask).sum(dim=2)+1e-5) / (traj_aug_mask.sum(dim=2)+1e-5)
            loss_simple = loss_simple_map.mean([1,2,3])
        else:
            loss_simple = loss_simple_map.mean([1, 2, 3, 4])

        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        if self.logvar.device is not self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        # logvar_t = self.logvar[t.item()].to(self.device) # device conflict when ddp shared
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # loss_vlb = self.get_loss(model_output, target, mean=False, last_only=self.chunk).mean(dim=(1, 2, 3, 4))
        loss_vlb = loss_simple

        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict


    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, mask=None, cat_v_to_w=True, **kwargs):
        
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 1
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        use_ddim = ddim_steps is not None
        log = dict()

        if kwargs["split"] == "val":
            self.rand_cond_frame = False

        batch_input_data = self.get_batch_input(batch, random_uncond=False,
                                                    return_first_stage_outputs=True,
                                                    return_original_cond=True,
                                                    return_fs=True,
                                                    return_did=True,
                                                    return_traj=True)
        self.rand_cond_frame = True

        z, cond, xrec, xc, fs, did, cond_x = batch_input_data[:7]
        

        traj = batch_input_data[-1]
        
        N = xrec.shape[0]
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        log["traj"] = traj

        kwargs.update({"fs": fs.long()})
        kwargs.update({"domain_id": did.long()})

        c_cat = None

        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(cond, dict):
                    c_emb = cond["c_crossattn"][0]
                    if 'c_concat' in cond.keys():
                        c_cat = cond["c_concat"]
                else:
                    c_emb = cond

                ### bv, c, h, w
                img = torch.zeros_like(xrec[:,:,0])
                uc_img_emb = self.embedder(img) ### bv, l, c
                uc_img_emb = self.image_proj_model(uc_img_emb)

                uc = torch.cat([uc_img_emb, c_emb[:, uc_img_emb.shape[1]:]], dim=1)

                u_c_cat =  deepcopy(c_cat)
                if self.use_cat_mask:
                    u_c_cat[1][:,:,-self.chunk:] = torch.zeros_like(u_c_cat[1][:,:,-self.chunk:])
                    u_c_cat[0] = u_c_cat[0]*u_c_cat[1]

                ## hybrid case
                if isinstance(cond, dict):
                    # c_cat contains 2 types of variables
                    uc_hybrid = {"c_concat": u_c_cat, "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                kwargs.update({"return_intermediates": plot_denoise_rows})
                samples, z_denoise_row = self.sample_log(
                    cond=cond, batch_size=N, ddim=use_ddim,
                    ddim_steps=ddim_steps,causal=True,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z,chunk=self.chunk,cat_mask=self.use_cat_mask,sparse=self.sparse_memory,traj=False,**kwargs)
            x_samples = self.decode_first_stage(samples.to(z.device)).data.cpu()
            
            if cat_v_to_w:
                x_samples = rearrange(x_samples, '(b v) c t h w -> b c t h (v w)', v=self.n_view)
            log["samples"] = x_samples

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log
    
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, causal=False, chunk=4, cat_mask=False, sparse=False, inference=False, ddim_dtype=None, **kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self, num_chunk=self.ddim_num_chunk, dtype=ddim_dtype)
            shape = (self.channels, self.n_view, self.temporal_length, *self.image_size)            
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, batch_size, shape, cond, verbose=False, causal=causal, chunk=chunk, cat_mask=cat_mask, sparse=sparse, inference=inference, **kwargs
            )
        else:
            samples, intermediates = self.sample(
                cond=cond, batch_size=batch_size, return_intermediates=True,causal=causal,chunk=chunk,cat_mask=cat_mask,sparse=sparse, **kwargs
            )

        return samples, intermediates


    @torch.no_grad()
    def inference(
        self, config, memories, action, delta_action,
        c2w_list, w2c_list, intrinsic_list, target_dir, num_chunk,
        n_previous=4, chunk=None, n_valid=-1,
        sample=True, ddim_steps=50, ddim_eta=1, unconditional_guidance_scale=7.5, guidance_rescale=0.7, dataset_name="agibotworld", saving_tag="",
        inference_dtype = torch.float16, fps=2,
        saving_video=False, saving_fps=5, video_dir=None,
        **kwargs
    ):

        fps = fps*torch.ones((1,)).to(self.device)
        domain_id = torch.LongTensor([DomainTable[dataset_name],])

        kwargs.update({"dtype": inference_dtype})
        kwargs.update({"timestep_spacing": "uniform_trailing"})
        kwargs.update({"guidance_rescale": guidance_rescale})
        kwargs.update({"return_intermediates": False})
        self.ddim_num_chunk = 1
        self.rand_cond_frame = False

        os.makedirs(target_dir, exist_ok=True)
        if saving_video:
            os.makedirs(video_dir, exist_ok=True)

        if chunk is None:
            chunk = self.chunk
        if n_previous is None:
            n_previous = config.data.params.train.params.n_previous
        
        sample_size = tuple(config.data.params.train.params.sample_size)
        trans_resize = transforms.Compose([
            transforms.Resize(sample_size),
        ])
        trans_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        if len(memories.shape) == 4:
            memories = memories.unsqueeze(1)
        
        _, _, _, h_ori, w_ori = memories.shape

        ### c, v, t, h, w
        video_list = rearrange(memories, "c v t h w -> (v t) c h w")
        video_list = trans_resize(video_list)
        video_list = trans_norm(video_list)
        video_list = rearrange(video_list, "(v t) c h w -> c v t h w", v=self.n_view)

        ### v, t, 4, 4
        if len(w2c_list.shape) == 3:
            w2c_list = w2c_list.unsqueeze(0)
        if len(c2w_list.shape) == 3:
            c2w_list = c2w_list.unsqueeze(0)

        ### v, 3, 3
        if len(intrinsic_list.shape) == 2:
            intrinsic_list = intrinsic_list.unsqueeze(0)
        h_scale = float(sample_size[0])/h_ori
        w_scale = float(sample_size[1])/w_ori
        intrinsic_list[:, 0, 0] *= w_scale
        intrinsic_list[:, 0, 2] *= w_scale
        intrinsic_list[:, 1, 1] *= h_scale
        intrinsic_list[:, 1, 2] *= h_scale

        action = action[:n_previous+num_chunk*chunk, :]
        delta_action = delta_action[:num_chunk*chunk, :]

        ### get traj.
        traj_list = self.get_traj(
            sample_size, action, w2c_list, c2w_list, intrinsic_list,
        )
        traj_list = rearrange(traj_list, "c v t h w -> (v t) c h w")
        traj_list = trans_norm(traj_list)
        traj_list = rearrange(traj_list, "(v t) c h w -> c v t h w", v=self.n_view)
        
        ### b, c, v, t, h, w
        video_list = video_list.unsqueeze(dim=0)
        ### b, c, v, t, h, w
        traj_list = traj_list.unsqueeze(dim=0)

        sidx = 0
        eidx = n_previous
        ori_video = video_list.clone()
        ori_traj = traj_list.clone()

        delta_action = delta_action.unsqueeze(0)
        ori_delta_action = delta_action.clone()
        idx_final_id = n_previous+chunk

        ### b,v,t,4,4
        c2w_list = torch.linalg.inv(w2c_list).unsqueeze(dim=0)
        ### b,v,3,3
        intrinsic_list = intrinsic_list.unsqueeze(dim=0)

        ### pred: the output video
        ###
        ### b,c,v,1,h,w
        pred = None
        traj_all = None
        all_x_samples = None
        all_samples = None
        all_trajs = None
        all_c2ws_list = None

        print("Start Prediction...")
        for i_chunk in tqdm(range(num_chunk)):

            print(i_chunk)

            if i_chunk == 0:
                video = torch.cat((
                    ori_video[:,:,:,:n_previous,:,:],
                    ori_video[:,:,:,n_previous-1:n_previous].repeat(1,1,1,chunk,1,1)
                ), dim=3)
            else:
                ### all_x_samples: b,c,v,t,h,w
                n_history = all_x_samples.shape[3]
                idx_history = [n_history*_//(n_previous-1) for _ in range(n_previous-1)]  ### 0, 1/3, 2/3, 
                video = torch.cat((
                    all_x_samples[:,:,:,idx_history],
                    (x_samples[:,:,:,-1:]).repeat(1,1,1,chunk+1,1,1)
                ), dim=3)


            if i_chunk == 0:
                traj = ori_traj[:,:,:,i_chunk*chunk:(i_chunk+1)*chunk+n_previous]
                i_delta_action = ori_delta_action[:,i_chunk*chunk:(i_chunk+1)*chunk]
                i_c2w_list = c2w_list[:,:,i_chunk*chunk:(i_chunk+1)*chunk+n_previous]

            else:

                traj = torch.cat((
                    all_trajs[:,:,:,idx_history,:,:],
                    ori_traj[:,:,:,i_chunk*chunk+n_previous-1:(i_chunk+1)*chunk+n_previous,:,:],
                ), dim=3)

                i_delta_action = ori_delta_action[:,i_chunk*chunk+n_previous-1:(i_chunk+1)*chunk+n_previous,:]

                i_c2w_list = torch.cat((
                    all_c2ws_list[:,:,idx_history,:,:],
                    c2w_list[:,:,i_chunk*chunk+n_previous-1:(i_chunk+1)*chunk+n_previous,:,:],
                ), dim=2)

            if traj.shape[3] < chunk+n_previous:
                ### pad with last frame action
                traj = torch.cat((
                    traj,
                    traj[:,:,:,-1:].repeat(1,1,1,chunk+n_previous-traj.shape[3],1,1)
                ), dim=3)
                i_delta_action = torch.cat((
                    i_delta_action,
                    torch.zeros(
                        [i_delta_action.shape[0], chunk-i_delta_action.shape[1], i_delta_action.shape[2]],
                        dtype=i_delta_action.dtype, device=i_delta_action.device
                    )
                ), dim=1)
                i_c2w_list = torch.cat((
                    i_c2w_list,
                    i_c2w_list[:,:,-1:].repeat(1,1,chunk+n_previous-i_c2w_list.shape[2],1,1)
                ), dim=2)

            video = torch.clamp(video, min=-1, max=1)
            traj = torch.clamp(traj, min=-1, max=1)
            batch = dict(
                video = video.to(dtype=inference_dtype, device=self.device),
                traj=traj.to(dtype=inference_dtype, device=self.device),
                delta_action=i_delta_action.to(dtype=inference_dtype, device=self.device),
                domain_id=domain_id.to(device=self.device),
                intrinsic=intrinsic_list, extrinsic=i_c2w_list, caption=[""],
                cond_id=torch.tensor([-n_previous-chunk]).to(dtype=torch.int64), device=self.device,
                fps=fps,
            )


            b, c, v, t, h, w = video.shape

            pre_z = None
            pre_img_emb = None

            if i_chunk != 0:
                pre_z = torch.cat((
                    all_samples[:,:,idx_history].to(samples.device),
                    samples[:,:,-1:].repeat(1,1,chunk+1,1,1),
                ), dim=2).to(dtype=inference_dtype, device=self.device)
            
            z, cond, xc, fs, did, img_emb = self.get_batch_input(
                batch, random_uncond=False, return_first_stage_outputs=False,
                return_original_cond=True, return_fs=True,
                return_did=True,
                return_traj=False,
                return_img_emb=True,
                pre_z = pre_z,
                pre_img_emb = pre_img_emb,
            )
            torch.cuda.empty_cache()

            N = z.shape[0]
            c_cat = None

            kwargs.update({"fs": fs.long()})
            kwargs.update({"domain_id": did.long()})

            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                c_emb = cond["c_crossattn"][0]
                if 'c_concat' in cond.keys():
                    c_cat = cond["c_concat"]
                img = torch.zeros_like(
                    video[:,:,:,0,:,:]).view(-1, c, h, w).float().to(z.device)
                uc_img_emb = self.embedder(img) 
                uc_img_emb = self.image_proj_model(uc_img_emb.to(dtype=z.dtype, device=z.device))
                uc = torch.cat([
                    uc_img_emb,
                    c_emb[:, uc_img_emb.shape[1]:],
                ], dim=1)

                u_c_cat =  deepcopy(c_cat)
                if self.use_cat_mask:
                    u_c_cat[1][:,:,-self.chunk:] = torch.zeros_like(u_c_cat[1][:,:,-self.chunk:]).to(inference_dtype)
                    u_c_cat[0] = u_c_cat[0]*u_c_cat[1].to(inference_dtype)
                uc = {"c_concat": u_c_cat, "c_crossattn": [uc, ]}
            else:
                uc = None

            for _c_cat in range(len(cond["c_concat"])):
                cond["c_concat"][_c_cat] = cond["c_concat"][_c_cat].to(dtype=inference_dtype)
            for _c_cro in range(len(cond["c_crossattn"])):
                cond["c_crossattn"][_c_cro] = cond["c_crossattn"][_c_cro].to(dtype=inference_dtype)


            samples, _ = self.sample_log(
                cond=cond, batch_size=N, ddim=True,
                ddim_steps=ddim_steps, causal=True, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc, x0=z.to(inference_dtype), chunk=self.chunk,
                cat_mask=self.use_cat_mask, sparse=self.sparse_memory,
                traj=False, ddim_dtype=torch.float16, **kwargs
            )

            ### （b v）c t h w
            x_samples = self.decode_first_stage(samples.to(z.device)).data.cpu()
            ### b c v t h w
            x_samples = rearrange(x_samples, "(b v) c t h w -> b c v t h w", v=self.n_view)

            if all_x_samples is None:
                all_x_samples = x_samples.data.cpu()
                all_samples = samples.data.cpu()
                all_c2ws_list = i_c2w_list.data.cpu()
                all_trajs = traj.data.cpu()

            else:
                all_x_samples = torch.cat((all_x_samples, x_samples[:,:,:,n_previous:,:,:].data.cpu()), dim=3)
                all_samples = torch.cat((all_samples, samples[:,:,n_previous:,:,:].data.cpu()), dim=2)
                all_c2ws_list = torch.cat((all_c2ws_list, i_c2w_list[:,:,n_previous:].data.cpu()), dim=2)
                all_trajs = torch.cat((all_trajs, traj.data.cpu()[:,:,:,n_previous:]), dim=3)

            x_samples_pd = (x_samples+1.0) / 2.0
            x_samples_pd = torch.clamp(x_samples_pd, 0, 1)

            if pred is not None:
                pred = torch.cat((pred, x_samples_pd[:,:,:,n_previous:]), dim=3)
                traj_all = torch.cat((traj_all, (traj[:,:,:,n_previous:]+1)/2), dim=3)
            else:
                pred = x_samples_pd[:,:,:,n_previous:].clone()
                traj_all = (traj[:,:,:,n_previous:].clone()+1)/2

            idx_final_id += chunk


        x_samples_video = rearrange(pred, 'b c v t h w -> t h (b v w) c') * 255
        x_samples_video = torch.round(x_samples_video).to(torch.uint8)

        traj_samples_video = rearrange(traj_all, 'b c v t h w -> t h (b v w) c') * 255
        traj_samples_video = torch.round(traj_samples_video).to(torch.uint8)

        ### save video
        ### outputs: {t, 2h, (1 v w), c}
        if saving_video:
            outputs = torch.cat((x_samples_video, traj_samples_video), dim=1).data.cpu().numpy()
            if n_valid>0:
                outputs = outputs[:n_valid]
            container = av.open(os.path.join(video_dir, f'outputs{saving_tag}.mp4'), "w")
            stream = container.add_stream('h264', rate=saving_fps)
            stream.width = outputs[0].shape[1]
            stream.height = outputs[0].shape[0]
            for frame_i in range(outputs.shape[0]):
                frame = av.VideoFrame.from_ndarray(outputs[frame_i], format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()

        ### save images
        ### x_samples_video: {t, h, w, c}
        
        assert(x_samples_video.shape[0]>=n_valid)
        x_samples_video = x_samples_video[:n_valid]
        x_samples_video = x_samples_video.data.cpu().numpy()
        for frame_i in range(x_samples_video.shape[0]):
            frame = x_samples_video[frame_i]
            cv2.imwrite(os.path.join(target_dir, "frame_{:05d}.jpg".format(frame_i)), frame[:,:,::-1])


        del pred, traj_all, samples
        torch.cuda.empty_cache()

        print(f'Successfully generated')

        return x_samples_video, traj_samples_video


    @torch.no_grad()
    def get_traj(self, sample_size, pose, w2c, c2w, intrinsic, radius=50):
        """
        this function takes camera info. and eef. poses as inputs, and outputs the trajectory maps.
        output traj map shape: (c, v, t, h, w)
        """        
        h, w = sample_size

        if isinstance(pose, np.ndarray):
            pose = torch.tensor(pose, dtype=torch.float32)
        
        ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32, device=pose.device).view(1,1,4,4).permute(0,1,3,2)

        pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7]).unsqueeze(0)
        pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15]).unsqueeze(0)
        
        ee2cam_l = torch.matmul(w2c, pose_l_mat)
        ee2cam_r = torch.matmul(w2c, pose_r_mat)

        cvt_matrix = torch.tensor(Gripper2EEFCvt, dtype=torch.float32, device=pose.device).view(1,1,4,4)
        ee2cam_l = torch.matmul(ee2cam_l, cvt_matrix)
        ee2cam_r = torch.matmul(ee2cam_r, cvt_matrix)
        
        pts_l = torch.matmul(ee2cam_l, ee_key_pts)
        pts_r = torch.matmul(ee2cam_r, ee_key_pts)
        
        intrinsic = intrinsic.unsqueeze(1)

        uvs_l = torch.matmul(intrinsic, pts_l[:,:,:3,:])
        uvs_l = (uvs_l / pts_l[:,:,2:3,:])[:,:,:2,:].permute(0,1,3,2).to(dtype=torch.int64)

        uvs_r = torch.matmul(intrinsic, pts_r[:,:,:3,:])
        uvs_r = (uvs_r / pts_r[:,:,2:3,:])[:,:,:2,:].permute(0,1,3,2).to(dtype=torch.int64)

        all_img_list = []
        for iv in range(w2c.shape[0]):
            
            img_list = []
            for i in range(pose.shape[0]):
                
                img = np.zeros((h, w, 3), dtype=np.uint8) + 50

                ###
                ### Gripper Range in AgiBotWorld < 120
                normalized_value_l = pose[i, 7].item() / 120
                normalized_value_r = pose[i, 15].item() / 120
                color_l = ColorMapLeft(normalized_value_l)[:3]  # Get RGB values
                color_r = ColorMapRight(normalized_value_r)[:3]  # Get RGB values
                color_l = tuple(int(c * 255) for c in color_l)
                color_r = tuple(int(c * 255) for c in color_r)

                i_coord_list = []
                for points, color, colors, lr_tag in zip([uvs_l[iv, i], uvs_r[iv, i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                    base = np.array(points[0])
                    if base[0]<0 or base[0]>=w or base[1]<0 or base[1]>=h:
                        continue
                    point = np.array(points[0][:2])
                    cv2.circle(img, tuple(point), radius, color, -1)
                    

                for points, color, colors, lr_tag in zip([uvs_l[iv, i], uvs_r[iv, i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                    base = np.array(points[0]) # points:[4,3]
                    if base[0]<0 or base[0]>=w or base[1]<0 or base[1]>=h:
                        continue
                    for i, point in enumerate(points):
                        point = np.array(point[:2])
                        if i == 0:
                            continue
                        else:
                            cv2.line(img, tuple(base), tuple(point), colors[i-1], 8)

                img_list.append(img/255.)
            img_list = np.stack(img_list, axis=0)
            all_img_list.append(img_list)

        all_img_list = np.stack(all_img_list, axis=0)
        all_img_list = rearrange(torch.tensor(all_img_list), "v t h w c -> c v t h w").float()

        return all_img_list


    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        if not self.unet_trainable_list:
            params = list(self.model.parameters())
        else:
            params = []
            confirm_trainable_names = []
            self.model.requires_grad_(False)
            for name, param in self.model.named_parameters():
                for trainable_module_name in self.unet_trainable_list:
                    if trainable_module_name in name:
                        confirm_trainable_names.append(name)
                        param.requires_grad = True
                        params.append(param)
                        break
        mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        if self.cond_stage_trainable:
            params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            params.extend(params_cond_stage)
        
        if self.image_proj_model_trainable:
            mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            params.extend(list(self.image_proj_model.parameters()))   

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer





class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None,
                c_adm=None, s=None, mask=None, **kwargs):
        # temporal_context = fps is foNone
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'hybrid':
            ## it is just right [b,c,t,h,w]: concatenate in channel dim
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, **kwargs)
        else:
            raise NotImplementedError()

        return out
