import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from rl.utils import prepare_one_obs


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute PR-AUC (Average Precision) manually without sklearn.
    
    Args:
        y_true: Binary labels (0 or 1)
        y_scores: Prediction scores/probabilities for positive class
        
    Returns:
        PR-AUC score (0.0 to 1.0)
    """
    if len(np.unique(y_true)) <= 1:
        return 0.0
    
    # Sort by score descending
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Compute precision and recall at each threshold
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (np.sum(y_true) + 1e-8)
    
    # Compute AUC using trapezoidal rule
    # Add point (0, 1) at the beginning for recall=0, precision=1
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    
    # Trapezoidal integration
    pr_auc = np.trapz(precision, recall)
    return float(pr_auc)


class RewardFrameDataset(Dataset):
    """
    Treat each frame in a saved sample as an independent training example.
    Sample file format follows generate_actor_critic_discrete_data.py output.
    Supports both single directory (str) and multiple directories (List[str]).
    """

    def __init__(self, data_dirs: List[str], cfg: Any, processor: Any, torch_dtype: torch.dtype):
        self.cfg = cfg
        self.processor = processor
        self.torch_dtype = torch_dtype

        self.items: List[Tuple[str, int, int]] = []  # (file_path, frame_idx, reward_bool)
        
        # Support both single directory (str) and multiple directories (List[str])
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                continue
            for fname in os.listdir(data_dir):
                if not fname.endswith(".pt"):
                    continue
                fpath = os.path.join(data_dir, fname)
                sample = torch.load(fpath, map_location='cpu')
                # video = sample["video"]  # (T, H, W, 3)
                mask = sample["mask"]    # (T,)
                last_rew = sample["reward"]
                assert np.all(mask == 1) 
                # frag_rew = np.zeros_like(mask)
                # frag_rew[-1] = last_rew
                # reward_bool = 1 if float(sample["reward"]) > 0 else 0
                # for idx, valid in enumerate(mask):
                #     if bool(valid):
                #         self.items.append((fpath, idx, frag_rew[idx]))
                self.items.append((fpath, len(mask)-1, last_rew))

        if len(self.items) == 0:
            raise ValueError(f"No frames found in {data_dirs}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        fpath, frame_idx, rew = self.items[index]
        sample = torch.load(fpath, map_location='cpu')
        frame = sample["video"][frame_idx]  # HWC uint8
        instruction = sample["instruction"]

        obs = {"full_image": frame}
        inputs = prepare_one_obs(self.cfg, self.processor, obs, instruction, self.torch_dtype)
        # Remove proprio if unused (avoid None in collate)
        if not self.cfg.use_proprio and ("proprio" in inputs) and (inputs["proprio"] is None):
            inputs.pop("proprio", None)
        return inputs, torch.tensor(rew, dtype=torch.long)


def _simple_pad_batch(pad_token_id, inputs_list):
    # inputs_list elements have: input_ids, attention_mask, labels, pixel_values
    max_len = max(it["input_ids"].size(1) for it in inputs_list)
    padded = []
    for it in inputs_list:
        # skip None entries (e.g., proprio when unused)
        it = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in it.items() if v is not None}
        cur_len = it["input_ids"].size(1)
        if cur_len < max_len:
            pad_amt = max_len - cur_len
            bsz = it["input_ids"].size(0)
            pad_ids = it["input_ids"].new_full((bsz, pad_amt), pad_token_id)
            pad_mask = it["attention_mask"].new_zeros((bsz, pad_amt))
            pad_labels = it["labels"].new_full((bsz, pad_amt), -100)
            it["input_ids"] = torch.cat([it["input_ids"], pad_ids], dim=1)
            it["attention_mask"] = torch.cat([it["attention_mask"], pad_mask], dim=1)
            it["labels"] = torch.cat([it["labels"], pad_labels], dim=1)
        padded.append(it)

    batch = {}
    keys = padded[0].keys()
    for k in keys:
        tensors = [it[k] for it in padded]
        batch[k] = torch.cat(tensors, dim=0)
    return batch


def make_collate_fn(pad_token_id):
    def _fn(batch):
        inputs_list, labels = zip(*batch)
        batch_inputs = _simple_pad_batch(pad_token_id, list(inputs_list))
        batch_labels = torch.stack(labels)
        return batch_inputs, batch_labels
    return _fn

