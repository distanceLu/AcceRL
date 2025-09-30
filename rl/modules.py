
import torch
import torch.nn as nn


class AttentionPoolHead(nn.Module):
    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.attn_pool = AttentionPool(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size))

    def forward(self, hidden_states: torch.Tensor, add_emb: torch.Tensor = None) -> torch.Tensor:
        pooled = self.attn_pool(hidden_states)
        if add_emb is not None:
            pooled = pooled + add_emb
        out = self.mlp(pooled)
        return out


class AttentionPool(nn.Module):
    """
    注意力池化头，用于从隐藏状态中提取动作相关的表示。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn_pool = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态，形状为 (B, num_tokens, D)。

        Returns:
            torch.Tensor: 池化后的表示，形状为 (B, D)。
        """
        # 1. 计算注意力分数
        scores = self.attn_pool(hidden_states)  # (B, num_tokens, 1)
        # 2. 应用softmax获取注意力权重
        weights = torch.softmax(scores, dim=-2)  # (B, num_tokens, 1)
        # 3. 加权平均得到池化表示
        pooled = torch.sum(weights * hidden_states, dim=-2)  # (B, D)
        return pooled