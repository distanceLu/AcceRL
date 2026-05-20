import os

os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODEL_PATH = "/cpfs01/luck_hub_cache/huggingface/hub/Qwen3-VL-8B-Instruct"


def main():
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    raw_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    prompts = []
    for text in raw_prompts:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]
        prompts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=1024,
        tensor_parallel_size=1,
        distributed_executor_backend="mp",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=128,
    )

    outputs = llm.generate(prompts, sampling_params)

    for raw_prompt, output in zip(raw_prompts, outputs):
        print("-" * 50)
        print(f"Prompt: {raw_prompt!r}")
        print(f"Generated text: {output.outputs[0].text!r}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()