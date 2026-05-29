import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import multiprocessing as mp
import subprocess
import sys
import tempfile


MODEL_PATH = "/mnt/data/lcx4/why_workspace/hf_cache/RynnBrain-8B"


def make_random_video(path: str, num_frames: int, width: int = 224, height: int = 224):
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 6.0, (width, height))

    for _ in range(num_frames):
        frame = np.random.randint(
            0, 256, size=(height, width, 3), dtype=np.uint8
        )
        writer.write(frame)

    writer.release()


def make_random_video_in_subprocess(path: str, num_frames: int):
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--make-video",
            path,
            "--num-frames",
            str(num_frames),
        ],
        check=True,
    )


def extract_video_frames(path: str, out_path: str, num_frames: int):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        indices = list(range(num_frames))
    elif total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()
    else:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx} from video: {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    np.save(out_path, np.stack(frames, axis=0))


def extract_video_frames_in_subprocess(path: str, out_path: str, num_frames: int):
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--extract-video",
            path,
            "--out-npy",
            out_path,
            "--num-frames",
            str(num_frames),
        ],
        check=True,
    )


def video_metadata(num_frames: int, fps: float = 6.0):
    return {
        "fps": fps,
        "frames_indices": list(range(num_frames)),
        "total_num_frames": num_frames,
        "duration": num_frames / fps,
    }


def main():
    import numpy as np
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    tmpdir = tempfile.mkdtemp(prefix="vllm_dual_video_demo_")
    original_video = os.path.join(tmpdir, "random_original.mp4")
    predicted_video = os.path.join(tmpdir, "random_predicted.mp4")
    original_npy = os.path.join(tmpdir, "random_original.npy")
    predicted_npy = os.path.join(tmpdir, "random_predicted.npy")

    make_random_video_in_subprocess(original_video, num_frames=16)
    make_random_video_in_subprocess(predicted_video, num_frames=3)
    extract_video_frames_in_subprocess(original_video, original_npy, num_frames=16)
    extract_video_frames_in_subprocess(predicted_video, predicted_npy, num_frames=3)

    original_frames = np.load(original_npy)
    predicted_frames = np.load(predicted_npy)

    # 把聊天信息转化成模型需要的prompt格式
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=40960,
        max_num_batched_tokens=40960,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 2},
        disable_log_stats=True,
    )

    question = """You are given two videos in order.
(1) The first video is the original observed egocentric clip.
(2) The second video is a predicted future continuation of the first video. It may be useful, but it can be imperfect.

Use the original video as primary evidence and the predicted future video as auxiliary evidence.
Question: what will the person do next after this video?
Answer with a short action phrase only. Do not output XML, tool calls, file paths, or explanations."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": original_video,
                    "nframes": 16,
                    "min_pixels": 16384,
                    "max_pixels": 200704,
                },
                {
                    "type": "video",
                    "video": predicted_video,
                    "nframes": 3,
                    "min_pixels": 16384,
                    "max_pixels": 200704,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]

    # 应用聊天模版将message转化成prompt
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "video": [
                (original_frames, video_metadata(16)),
                (predicted_frames, video_metadata(3)),
            ],
        },
    }

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
    )

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    prediction = outputs[0].outputs[0].text.strip()

    print("original_video:", original_video)
    print("predicted_video:", predicted_video)
    print("prediction:", prediction)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-video")
    parser.add_argument("--extract-video")
    parser.add_argument("--out-npy")
    parser.add_argument("--num-frames", type=int)
    args = parser.parse_args()

    if args.make_video:
        make_random_video(args.make_video, num_frames=args.num_frames)
    elif args.extract_video:
        extract_video_frames(
            args.extract_video,
            out_path=args.out_npy,
            num_frames=args.num_frames,
        )
    else:
        main()
