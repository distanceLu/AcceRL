import json
import numpy as np
import h5py
import cv2
from moviepy.editor import VideoFileClip
import argparse
import json
import os


def extract_info(data_root, task_id, episode_id, slices, save_root, cam_name):
    """
    Convertion from the original agibotworld data format to the expected inference data format
    """
    video_reader = VideoFileClip(os.path.join(data_root, "observations", task_id, episode_id, "videos", cam_name+"_color.mp4"))
    ### agibotworld videos are 30fps
    frame = video_reader.get_frame(float(slices[0])/30.0)
    cv2.imwrite(os.path.join(save_root, "frame.png"), frame[:,:,::-1])
    video_reader.close()


    with h5py.File(os.path.join(data_root, "proprio_stats", task_id, episode_id, "proprio_stats.h5"), "r") as fid:
        all_abs_gripper = np.array(fid[f"state/effector/position"], dtype=np.float32)
        print(len(all_abs_gripper))
        all_abs_gripper = all_abs_gripper[slices]
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)[slices]
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)[slices]
        

    data = np.zeros([len(slices), 16])
    data[:, 0:3] = all_ends_p[:, 0]
    data[:, 8:11] = all_ends_p[:, 1]
    data[:, 3:7] = all_ends_o[:, 0]
    data[:, 11:15] = all_ends_o[:, 1]
    data[:, 7] = all_abs_gripper[:, 0]
    data[:, 15] = all_abs_gripper[:, 1]
    np.save(os.path.join(save_root, "action.npy"), data)


    with open(os.path.join(data_root, "parameters", task_id, episode_id, "parameters", "camera", cam_name+"_intrinsic_params.json"), "r") as f:
        info = json.load(f)["intrinsic"]
    K = np.eye(3)
    K[0,0] = info["fx"]
    K[1,1] = info["fy"]
    K[0,2] = info["ppx"]
    K[1,2] = info["ppy"]
    np.save(os.path.join(save_root, "intrinsics.npy"), K)

    with open(os.path.join(data_root, "parameters", task_id, episode_id, "parameters", "camera", cam_name+"_extrinsic_params_aligned.json"), "r") as f:
        info = json.load(f)
    c2w = np.eye(4)
    c2w[:3, :3] = info[slices[0]]["extrinsic"]["rotation_matrix"]
    c2w[:3, 3] = info[slices[0]]["extrinsic"]["translation_vector"]
    np.save(os.path.join(save_root, "extrinsics.npy"), c2w)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="help document")

    parser.add_argument(
        "--data_root", "-r", type=str,
        help="Path to the root of agibotworld dataset"
    )
    parser.add_argument(
        "--task_id", "-t", type=str,
        help="taskid in agibotworld"
    )
    parser.add_argument(
        "--episode_id", "-e", type=str,
        help="episode id in agibotworld"
    )
    parser.add_argument(
        "--save_root", "-s", type=str,
    )
    parser.add_argument(
        "--index", "-i", type=int, nargs='+', default=None,
        help="action index to extract, usage: -i 0 1 2"
    )
    parser.add_argument(
        "--index_json", "-j", type=str, default=None,
        help="action index json"
    )
    parser.add_argument(
        "--cam_name", "-c", type=str, default="head"
    )
    args = parser.parse_args()

    if args.index_json is not None:
        with open(args.index_json, "r") as f:
            index = json.load(f)
    elif args.index is not None:
        index = args.index
    else:
        raise ValueError("Neighter index_json nor index is provided.")

    os.makedirs(args.save_root, exist_ok=True)

    extract_info(
        args.data_root, args.task_id, args.episode_id, index, args.save_root, args.cam_name
    )

