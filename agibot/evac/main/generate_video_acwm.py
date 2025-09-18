import sys
import os
import torch
import numpy as np
import math
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.general_utils import load_checkpoints, instantiate_from_config
from lvdm.data.get_actions import get_actions, parse_h5
from lvdm.data.statistics import StatisticInfo


def load_model(config):
    model = instantiate_from_config(config.model)
    model = load_checkpoints(model, config.model, ignore_mismatched_sizes=False)
    return model


def load_config(args):
    config_file = args.config_path
    config = OmegaConf.load(config_file)
    config.model.pretrained_checkpoint = args.ckp_path
    return config


def get_image(img_path, n):
    img = np.array(Image.open(img_path))
    img = torch.from_numpy(img).float().permute(2,0,1)/255.0
    img = img.unsqueeze(1).repeat(1,n,1,1)
    return img

def get_action_bias_std(domain_name):
    return torch.tensor(StatisticInfo[domain_name]['mean']).unsqueeze(0), torch.tensor(StatisticInfo[domain_name]['std']).unsqueeze(0)

def get_action_npy(
    action_path, n_chunk, chunk, n_previous, sep=1, domain_name="agibotworld"
):
    abs_act = np.load(action_path)

    if n_chunk > 0:
        if (abs_act.shape[0] < n_chunk * chunk + n_previous):
            raise ValueError(f"Num of Action Timestep {abs_act.shape[0]} smaller than {n_previous}+{n_chunk}*{n_chunk}")
        assert(abs_act.shape[1] == 16)
        abs_act = abs_act[:n_chunk*chunk+n_previous, :]

    action, delta_action = get_actions(
        gripper=np.stack((abs_act[:, 7], abs_act[:, 15]), axis=1),
        all_ends_p=np.stack((abs_act[:, 0:3], abs_act[:, 8:11]), axis=1),
        all_ends_o=np.stack((abs_act[:, 3:7], abs_act[:, 11:15]), axis=1),
        slices=None,
        delta_act_sidx=n_previous,
    )
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    delta_act_meanv, delta_act_stdv = get_action_bias_std(domain_name)

    delta_action[:, :6] = (delta_action[:, :6] - sep*delta_act_meanv[:, :6]) / (sep*delta_act_stdv[:, :6])
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep*delta_act_meanv[:, 6:]) / (sep*delta_act_stdv[:, 6:])
    return action, delta_action


def get_action_h5(
    action_path, n_chunk, chunk, n_previous, sep=1, domain_name="agibotworld"
):
    if n_chunk > 0:
        slices = list(range(0, n_chunk*chunk+n_previous))
    else:
        slices = None
    action, delta_action = parse_h5(action_path, slices=slices, delta_act_sidx=n_previous) 
    action = torch.FloatTensor(action)
    delta_action = torch.FloatTensor(delta_action)
    delta_act_meanv, delta_act_stdv = get_action_bias_std(domain_name)
    delta_action[:, :6] = (delta_action[:, :6] - sep*delta_act_meanv[:, :6]) / (sep*delta_act_stdv[:, :6])
    delta_action[:, 7:13] = (delta_action[:, 7:13] - sep*delta_act_meanv[:, 6:]) / (sep*delta_act_stdv[:, 6:])
    return action, delta_action


def get_caminfo_npy(extrinsic_path, intrinsic_path, n):
    c2w = torch.from_numpy(np.load(extrinsic_path))
    w2c = torch.linalg.inv(c2w).float()
    intrinsic = torch.from_numpy(np.load(intrinsic_path)).float()
    w2c = w2c.unsqueeze(0).repeat(n,1,1)
    c2w = c2w.unsqueeze(0).repeat(n,1,1)
    return c2w, w2c, intrinsic

def get_caminfo_json(extrinsic_path, intrinsic_path, n):
    with open(extrinsic_path, "r") as f:
        info = json.load(f)[0]
        c2w = np.eye(4)
        c2w[:3,:3] = np.array(info["extrinsic"]["rotation_matrix"])
        c2w[:3, 3] = np.array(info["extrinsic"]["translation_vector"])
    c2w = torch.from_numpy(c2w).float()
    w2c = torch.linalg.inv(c2w).float()
    w2c = w2c.unsqueeze(0).repeat(n,1,1)
    c2w = c2w.unsqueeze(0).repeat(n,1,1)
    with open(intrinsic_path, "r") as f:
        info = json.load(f)["intrinsic"]
    intrinsic = np.eye(3)
    intrinsic[0,0] = info["fx"]
    intrinsic[0,2] = info["ppx"]
    intrinsic[1,1] = info["fy"]
    intrinsic[1,2] = info["ppy"]
    intrinsic = torch.from_numpy(intrinsic).float()
    return c2w, w2c, intrinsic


def main(args):

    seed_everything(args.seed)
    device = torch.device(args.device)

    ### load config
    config = load_config(args)

    chunk = config.chunk
    n_previous = config.n_previous

    ### 
    img = get_image(
        args.input_path, n_previous
    )

    ###
    if args.action_path.endswith(".h5"):
        action, delta_action = get_action_h5(
            args.action_path, args.n_chunk, chunk, n_previous,
            sep=config.data.params.train.params.max_sep,
            domain_name="agibotworld"
        )
    elif args.action_path.endswith(".npy"):
        action, delta_action = get_action_npy(
            args.action_path, args.n_chunk, chunk, n_previous,
            sep=config.data.params.train.params.max_sep,
            domain_name="agibotworld"
        )
    else:
        raise NotImplementedError

    n = action.shape[0]

    ###
    if args.extrinsic_path.endswith(".json"):
        c2w, w2c, intrinsic = get_caminfo_json(
            args.extrinsic_path,
            args.intrinsic_path,
            n
        )

    elif args.extrinsic_path.endswith(".npy"):
        c2w, w2c, intrinsic = get_caminfo_npy(
            args.extrinsic_path,
            args.intrinsic_path,
            n
        )
    else:
        raise NotImplementedError


    ###
    model = load_model(config).to(device=device)
    model.eval()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        model.inference(
            config, img, action, delta_action,
            c2w, w2c, intrinsic,
            args.save_root, int(math.ceil((float(n)-n_previous)/chunk)),
            chunk=chunk, n_previous=n_previous, n_valid=n-n_previous,
            unconditional_guidance_scale=args.cfg,
            guidanc_erescale=args.gr,
            ddim_steps=args.ddim_steps, 
            saving_tag="", saving_fps=5,
            saving_video=True,
            video_dir=os.path.join(args.save_root, "video")
        )
        torch.cuda.empty_cache()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="help document")

    parser.add_argument(
        "--input_path", "-i", type=str,
        help="Path to the input image file",
        default="examples/examples0/frame.png"
    )
    parser.add_argument(
        "--action_path", "-a", type=str,
        help="Path to the .npy or .h5 file including the ABSOLUTE actions of end-effector. The file should contain a {T x 16} numpy array: T x [xyz_left(3), quat_xyzw_left(4), gripper_left(1), xyz_right(3), quat_xyzw_right(4), gripper_right(1)]",
        default="examples/examples0/action.npy"
    )
    parser.add_argument(
        "--extrinsic_path", "-ex", type=str,
        help="Path to the .npy or .json file of camera extrinsics {4 x 4}",
        default="examples/examples0/extrinsics.npy"
    )
    parser.add_argument(
        "--intrinsic_path", "-in", type=str,
        help="Path to the .npy or .json file of camera intrinsics {3 x 3}",
        default="examples/examples0/intrinsics.npy"
    )
    parser.add_argument(
        "--save_root", "-s", type=str,
        help="Path to save predictions",
        default="./result_video_root/examples0"
    )
    parser.add_argument(
        "--ckp_path", type=str,
        default="./model/EnerV_AC_deepspeed_v0.1.pt"
    )
    parser.add_argument(
        "--config_path", type=str,
        default="./configs/agibotworld/config.yaml"
    )

    parser.add_argument(
        "--n_chunk", type=int, default=-1,
        help="number of chunks to predict"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=27,
    )
    parser.add_argument(
        "--cfg", type=float, default=1,
        help="unconditional guidance scale ",
    )
    parser.add_argument(
        "--gr", type=float, default=0.7,
        help="guidance rescale",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda:0"
    )
    parser.add_argument(
        "--seed", type=int,
        default=12345
    )

    args = parser.parse_args()

    main(args)