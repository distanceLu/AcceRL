# SPDX-License-Identifier: Apache-2.0
"""Run local vLLM inference from a local model directory."""

from vllm import LLM, SamplingParams

MODEL_NAME = "/mnt/data/lcx4/hf_cache/RynnBrain-8B"


def create_llm() -> LLM:
    """Create a vLLM engine for 4-GPU local inference.

    vllm.LLM does not support multi-replica data parallelism in a single
    Python process, so this script uses tensor parallelism across the 4
    visible GPUs.
    """
    return LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        gpu_memory_utilization=0.7,

        # Keep vLLM's initialization/profile dummy batch small for this demo.
        # The default can be 16K tokens, which is very slow for single-GPU MoE.
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        disable_custom_all_reduce=True,
        max_model_len=2048,
    )

    # Single-GPU fallback if DP/EP is not needed or the 4-GPU config fails:
    # return LLM(
    #     model=MODEL_NAME,
    #     trust_remote_code=True,
    #     dtype="bfloat16",
    #     enforce_eager=True,
    #     gpu_memory_utilization=0.7,
    # )

    # return LLM(
    #     model=MODEL_NAME,
    #     trust_remote_code=True,
    #     dtype="bfloat16",
    #     enforce_eager=True,
    #     tensor_parallel_size=1,
    #     gpu_memory_utilization=0.7,
    #     max_model_len=2048,
    #     # Keep vLLM's initialization/profile dummy batch small for this demo.
    #     # The default can be 16K tokens, which is very slow for single-GPU MoE.
    #     max_num_batched_tokens=1024,
    #     max_num_seqs=4,
    # )


def main() -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    print(f"[init] Loading local model from {MODEL_NAME}")
    llm = create_llm()

    print("[generate] Starting generation...")
    outputs = llm.generate(prompts, sampling_params)
    print("[generate] Generation complete.")

    print("-" * 60)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
