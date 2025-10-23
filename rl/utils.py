from experiments.robot.openvla_utils import prepare_images_for_vla, normalize_proprio
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, IGNORE_INDEX, ACTION_DIM
import torch
import numpy as np
from typing import Any, Dict, List, Tuple
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    STOP_INDEX,
    ACTION_TOKEN_BEGIN_IDX
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from transformers.modeling_outputs import CausalLMOutputWithPast
from contextlib import nullcontext
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file
from pathlib import Path
from peft import PeftModel


def normalize_proprio_batch(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Proprioception data, shape (B, D)
        norm_stats: Normalization statistics. Expected keys:
            - For BOUNDS: "min", "max" (+ optional "mask")
            - For BOUNDS_Q99: "q01", "q99" (+ optional "mask")

    Returns:
        np.ndarray: Normalized proprioception data, shape (B, D)
    """
    proprio = np.asarray(proprio)
    if proprio.ndim != 2:
        raise ValueError(f"proprio must have shape (B, D), got {proprio.shape}")
    B, D = proprio.shape
    dtype = proprio.dtype

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        low = np.asarray(norm_stats["min"], dtype=dtype)
        high = np.asarray(norm_stats["max"], dtype=dtype)
        mask = np.asarray(norm_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        low = np.asarray(norm_stats["q01"], dtype=dtype)
        high = np.asarray(norm_stats["q99"], dtype=dtype)
        mask = np.asarray(norm_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    # Validate stat shapes
    if low.ndim != 1 or high.ndim != 1 or low.shape[0] != D or high.shape[0] != D:
        raise ValueError(f"Normalization stats must be 1D of length D={D}. "
                         f"Got low={low.shape}, high={high.shape}")
    if mask.ndim != 1 or mask.shape[0] != D:
        raise ValueError(f"mask must be 1D of length D={D}, got {mask.shape}")

    # Broadcast to (B, D)
    low = low.reshape(1, D)
    high = high.reshape(1, D)
    mask = mask.reshape(1, D)

    normalized = np.clip(
        np.where(
            mask,
            2.0 * (proprio - low) / (high - low + np.asarray(1e-8, dtype=dtype)) - 1.0,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    return normalized


def prepare_one_obs(
    cfg: Any,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    torch_dtype: torch.dtype,
) -> Dict:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
    """
    # Collect all input images
    all_images = [obs["full_image"]]
    if cfg.num_images_in_input > 1:
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

    # Process images
    all_images = prepare_images_for_vla(all_images, cfg)

    # Extract primary image and additional images
    primary_image = all_images.pop(0)

    # Build VLA prompt
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    
    # Process primary image
    inputs = processor(prompt, primary_image).to(dtype=torch_dtype)

    # Process additional wrist images if any
    if all_images:
        all_wrist_inputs = [
            processor(prompt, image_wrist).to(dtype=torch_dtype) for image_wrist in all_images
        ]
        # Concatenate all images
        primary_pixel_values = inputs["pixel_values"]
        all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
        inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)
    # Process proprioception data if used
    proprio = None
    if cfg.use_proprio:
        proprio = obs["state"]
    """
    本样本注释可以方便代码理解和修改，不要删除
    inputs example:
    input_ids torch.Size([1, 34])
    attention_mask torch.Size([1, 34])
    pixel_values torch.Size([1, 12, 224, 224])

    proprio example: (8,), numpy.ndarray
    """
    input_ids, attention_mask, labels = process_one_obs(
        inputs["input_ids"], inputs["attention_mask"]
    )
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    inputs["labels"] = labels
    inputs["proprio"] = proprio
    return inputs


def process_one_obs(input_ids, attention_mask):
    assert input_ids.shape[0] == 1 and attention_mask.shape[0] == 1, "Only batch size 1 is supported for now"
    # If the special empty token ('') does not already appear after the colon (':') token in the prompt
    # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )

    # Create fake labels tensor (needed for action mask)
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX

    # Prepare inputs by adding necessary tokens
    input_ids, attention_mask = prepare_input_for_action_prediction(input_ids, attention_mask)

    # Update labels tensor for action mask computation later
    labels = prepare_labels_for_action_prediction(labels, input_ids)

    return input_ids, attention_mask, labels


def prepare_input_for_action_prediction(input_ids, attention_mask):
    """Prepares input for action prediction by adding necessary tokens"""
    # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
    placeholder_action_token_ids = (
        torch.ones((input_ids.shape[0], ACTION_DIM * NUM_ACTIONS_CHUNK)).to(input_ids.device).to(input_ids.dtype)
    )
    input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

    # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
    stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
    input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

    # Extend the attention mask to fit the new shape of input
    # Note: Only batch size == 1 supported right now
    mask_extension = (
        torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
        .to(attention_mask.device)
        .to(attention_mask.dtype)
    )
    attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

    return input_ids, attention_mask


def prepare_labels_for_action_prediction(labels, input_ids):
    """Creates labels tensor for action prediction if not provided"""
    # Extend labels tensor with fake action labels
    ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
    labels_extension = (
        torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
        * ARBITRARY_ACTION_TOKEN_IDX
    )
    labels = torch.cat([labels, labels_extension], dim=-1)

    # Replace last label token with stop token
    labels[:, -1] = STOP_INDEX

    return labels
    

def _autocast_ctx(device: torch.device, torch_dtype):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch_dtype)
    return nullcontext()


def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device: torch.device,
    use_l1_regression,
    use_proprio,
    use_film,
    torch_dtype: torch.dtype
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics.
    """
    metrics = {}

    # VLA forward pass
    with _autocast_ctx(device, torch_dtype):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch_dtype).to(device),
            labels=batch["labels"],  # HF内部会处理 dtype/cast
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
        )

    # Get masks for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if use_proprio:
        num_patches += 1

    # Discrete (next-token) vs continuous (L1/diffusion)
    if not use_l1_regression:
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    else:
        # Continuous action head path
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch_dtype)
        )  # (B, act_chunk_len, D)

        if use_l1_regression:
            predicted_actions = action_head.predict_action(actions_hidden_states)

    return predicted_actions


def batch_process_obs(
    vla,
    inputs_list: List[Dict[str, Any]],
):
    # 目标序列最大长度（对齐到同一个 max_len，确保各 key 同长）
    max_len = max(it["input_ids"].size(1) for it in inputs_list)
    pad_id = int(vla.pad_token_id)  # 例如 Llama 的 <pad>，若无请在模型配置中设置

    # 对每条样本进行右侧 padding
    for it in inputs_list:
        cur_len = it["input_ids"].size(1)
        if cur_len < max_len:
            pad_amt = max_len - cur_len
            bsz = it["input_ids"].size(0)  # 通常为 1

            # input_ids: 用 pad_id
            pad_ids = it["input_ids"].new_full((bsz, pad_amt), pad_id)
            it["input_ids"] = torch.cat([it["input_ids"], pad_ids], dim=1)

            # attention_mask: 用 0
            pad_mask = it["attention_mask"].new_zeros((bsz, pad_amt))
            it["attention_mask"] = torch.cat([it["attention_mask"], pad_mask], dim=1)

            # labels: 用 -100
            pad_labels = it["labels"].new_full((bsz, pad_amt), -100)
            it["labels"] = torch.cat([it["labels"], pad_labels], dim=1)

    # 聚合成 batch，并移动到目标设备
    inputs = {}
    keys = inputs_list[0].keys()
    for k in keys:
        tensors = [it[k] for it in inputs_list]
        inputs[k] = torch.cat(tensors, dim=0).to(vla.device)
    inputs['proprio'] = inputs['proprio'].to(torch.float32)
    return inputs


def my_get_action(vla, cfg, processor, observations, action_head, proprio_projector, torch_dtype: torch.dtype):
    inputs_list = []
    for obs in observations:
        inputs_t = prepare_one_obs(cfg, processor, obs, obs["task_description"], torch_dtype)
        inputs_list.append(inputs_t)
    for inputs_t in inputs_list:
        proprio_t_norm = normalize_proprio(inputs_t['proprio'], vla.norm_stats[cfg.unnorm_key]["proprio"])
        inputs_t["proprio"] = torch.tensor(proprio_t_norm)
        # 基本一致性检查（单条样本内长度应一致）
        assert inputs_t["input_ids"].size(1) == inputs_t["attention_mask"].size(1) == inputs_t["labels"].size(1), \
            "Per-sample sequence lengths of input_ids/attention_mask/labels must match."
    inputs_batch = batch_process_obs(vla, inputs_list)
    norm_action = run_forward_pass(
        vla=vla,
        action_head=action_head,
        proprio_projector=proprio_projector,
        batch=inputs_batch,
        action_tokenizer=None,
        device=vla.device,
        use_l1_regression=cfg.use_l1_regression,
        use_proprio=cfg.use_proprio,
        use_film=cfg.use_film,
        torch_dtype=torch_dtype,
    )
    actions = vla._unnormalize_actions(norm_action[0].float().detach().cpu().numpy(), cfg.unnorm_key)
    return actions


def check_unnorm_key(cfg, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.unnorm_key
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"


def get_vla(cfg: Any, torch_dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """
    只读加载 OpenVLA：不修改 checkpoint 内的 config.json。
    """
    print("Instantiating pretrained VLA policy (read-only, no config.json mutation)...")

    # 1) 显式加载 Config（不会触发 auto_map 也不会写文件）
    vla_cfg = OpenVLAConfig.from_pretrained(
        cfg.pretrained_checkpoint,
        trust_remote_code=True,   # 允许自定义类
    )

    # 2) 显式加载模型（不走 Auto*，不需要 auto_map）
    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        config=vla_cfg,
        torch_dtype=torch_dtype,     #bfloat16 
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(cfg.device)

    # 3) FiLM（若启用）
    if getattr(cfg, "use_film", False):
        from experiments.robot.openvla_utils import _apply_film_to_vla
        vla = _apply_film_to_vla(vla, cfg)

    # 4) 设定输入图像数量
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # 5) 未量化时放到目标设备
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(cfg.device)

    # 6) 加载数据集统计（归一化/反归一化用）
    from experiments.robot.openvla_utils import _load_dataset_stats
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


def normalize_proprio(norm_stats, proprio: Any) -> np.ndarray:
    """
    Normalize proprioception data using self.vla.norm_stats[self.cfg.unnorm_key]["proprio"].
    Accepts numpy array or torch tensor; returns numpy array in [-1, 1].
    """
    # Convert to numpy
    if isinstance(proprio, torch.Tensor):
        proprio = proprio.detach().cpu().numpy()
    else:
        proprio = np.asarray(proprio)

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    return normalized_proprio


def batch_process_obs(pad_id, inputs_list: List[Dict[str, Any]], device, max_len=None) -> Dict[str, torch.Tensor]:
    """
    Right-pad variable-length sequences across a list of samples and stack into a batch on self.vla.device.
    Expects each item to contain: input_ids, attention_mask, labels, pixel_values, proprio, etc.
    """
    # 目标序列最大长度（对齐到同一个 max_len，确保各 key 同长）
    max_len_t = max(it["input_ids"].size(1) for it in inputs_list)
    if max_len:
        if max_len_t > max_len:
            print(f"Warning! input_ids size: {max_len_t}, max_len: {max_len}")
    else:
        max_len = max_len_t

    # 对每条样本进行右侧 padding
    for it in inputs_list:
        cur_len = it["input_ids"].size(1)
        if cur_len < max_len:
            pad_amt = max_len - cur_len
            bsz = it["input_ids"].size(0)  # 通常为 1

            # input_ids: pad_id
            pad_ids = it["input_ids"].new_full((bsz, pad_amt), pad_id)
            it["input_ids"] = torch.cat([it["input_ids"], pad_ids], dim=1)

            # attention_mask: 0
            pad_mask = it["attention_mask"].new_zeros((bsz, pad_amt))
            it["attention_mask"] = torch.cat([it["attention_mask"], pad_mask], dim=1)

            # labels: -100
            pad_labels = it["labels"].new_full((bsz, pad_amt), -100)
            it["labels"] = torch.cat([it["labels"], pad_labels], dim=1)

    # 聚合成 batch，并移动到目标设备
    inputs: Dict[str, torch.Tensor] = {}
    keys = inputs_list[0].keys()
    for k in keys:
        tensors = [it[k] for it in inputs_list]
        inputs[k] = torch.cat(tensors, dim=0).to(device)
    # inputs["proprio"] = inputs["proprio"].to(torch.float32)
    return inputs


def prepare_inputs_batch(model, inputs_list: List[Dict[str, Any]], max_len=None) -> Dict[str, torch.Tensor]:
    """
    对多条样本执行：
        - 归一化 proprio 到 [-1, 1]
        - 基本一致性检查
        - 序列右侧 padding 并拼 batch
    """
    inputs_list = inputs_list.copy()
    for i, it in enumerate(inputs_list):
        inputs_list[i] = it.copy()
    # Normalize proprio for each sample and run per-sample checks
    norm_stats = model.get_norm_stats()
    for it in inputs_list:
        # Normalize proprio using internal norm stats
        proprio_norm = normalize_proprio(norm_stats, it["proprio"])
        it["proprio"] = torch.tensor(proprio_norm, dtype=torch.float32).unsqueeze(dim=0)

        # Consistency check
        assert it["input_ids"].size(1) == it["attention_mask"].size(1) == it["labels"].size(1), \
            "Per-sample sequence lengths of input_ids/attention_mask/labels must match."

    # Batchify
    pad_id = int(model.vla.pad_token_id)
    return batch_process_obs(pad_id, inputs_list, model.device, max_len)


def compute_num_patches(vla, cfg) -> int:
    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        num_patches += 1
    return num_patches


def forward_vla(model, batch: Dict[str, torch.Tensor]):
    """
    Single VLA forward that returns output with hidden states.
    """
    ctx = torch.autocast("cuda", dtype=model.model_dtype) if model.device.type == "cuda" else nullcontext()
    with ctx:
        model.vla: OpenVLAForActionPrediction
        output = model.vla.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"].to(model.model_dtype),
            labels=batch["labels"],  # for mask derivation and potential loss
            output_hidden_states=True,
            proprio=batch["proprio"].to(model.model_dtype) if model.cfg.use_proprio else None,
            proprio_projector=model.proprio_projector if model.cfg.use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=model.cfg.use_film,
            this_act_emb=batch.get("this_act_emb", None),  # (B, 1, 4096) or None
            use_llm_loss=False,
        )
    return output


def load_lora_inplace(peft_model: PeftModel, lora_dir: Path):
    st_path = lora_dir / "adapter_model.safetensors"
    if st_path.exists():
        sd = load_file(str(st_path), device="cpu")
    else:
        sd = torch.load(lora_dir / "adapter_model.bin", map_location="cpu")
    set_peft_model_state_dict(peft_model, sd)  # 就地覆盖权重
    peft_model.set_adapter("default")
    peft_model.print_trainable_parameters()


def freeze_models(models):
    for model in models:
        for para in model.parameters():
            para.requires_grad = False