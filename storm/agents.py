import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from torch.cuda.amp import autocast

from storm.functions_losses import SymLogTwoHotLoss
from storm.utils import EMAScalar


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device="cuda")
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef, dist) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.dist = dist
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)
        self._max_std = 1.0
        self._min_std = 0.1
        self.action_dim = action_dim

        actor = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
        self.actor = nn.Sequential(
            *actor,
            nn.Linear(hidden_dim, action_dim*2)
        )

        critic = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])

        self.critic = nn.Sequential(
            *critic,
            nn.Linear(hidden_dim, 255)
        )
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    # def policy(self, x):
    #     logits = self.actor(x)
    #     return logits

    def policy(self, x):
        # print('x', x.shape)
        output = self.actor(x)
        mean = output[..., :self.action_dim]
        std = output[..., self.action_dim:]
        return mean, std

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        mean, std = self.policy(x)
        raw_value = self.critic(x)
        return mean, std, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            mean, std = self.policy(latent)
            if self.dist == "onehot":
                dist = distributions.Categorical(logits=mean)
                if greedy:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
            else:
                std = (self._max_std - self._min_std) * torch.sigmoid(
                    std + 2.0
                ) + self._min_std
                dist = distributions.normal.Normal(torch.tanh(mean), std)
                if greedy:
                    action = torch.tanh(mean)
                else:
                    action = dist.sample()
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        if self.dist == "onehot":
            return action.detach().cpu().squeeze(-1).numpy()
        else:
            return action.detach().cpu().squeeze(-2).float().numpy()

    def update(self, latent, action, old_logprob, old_value, reward, termination, train_steps, logger=None):
        '''
        Update policy and value model
        '''
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            mean, std, raw_value = self.get_logits_raw_value(latent)
            if self.dist == "onehot":
                dist = distributions.Categorical(logits=mean[:, :-1])
                log_prob = dist.log_prob(action)
            else:
                std = (self._max_std - self._min_std) * torch.sigmoid(
                    std + 2.0
                ) + self._min_std
                dist = distributions.normal.Normal(torch.tanh(mean[:, :-1]), std[:, :-1])
                log_prob = torch.sum(dist.log_prob(action), dim=-1)

            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).cuda(), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item(), train_steps)
            logger.log('ActorCritic/value_loss', value_loss.item(), train_steps)
            logger.log('ActorCritic/entropy_loss', entropy_loss.item(), train_steps)
            logger.log('ActorCritic/S', S.item(), train_steps)
            logger.log('ActorCritic/norm_ratio', norm_ratio.item(), train_steps)
            logger.log('ActorCritic/total_loss', loss.item(), train_steps)
