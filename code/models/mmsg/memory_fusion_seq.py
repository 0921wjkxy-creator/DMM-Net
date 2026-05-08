import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAttentionLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 2, dim_feedforward: int = None, dropout: float = 0.05):
        super().__init__()
        self.d_model = d_model

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward
        hidden = dim_feedforward or (2 * d_model)
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)

        # Norm + Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, curr, memory, curr_pos=None, memory_pos=None):
        # Self-Attention（在 curr 上）
        x = self.norm1(curr)
        if curr_pos is not None:
            x = x + curr_pos
        x, _ = self.self_attn(x, x, x)
        curr = curr + self.dropout(x)

        x = self.norm2(curr)
        mem = memory
        if memory_pos is not None:
            mem = mem + memory_pos
        x, _ = self.cross_attn(x, mem, mem)
        curr = curr + self.dropout(x)

        # Feed-forward
        x = self.norm3(curr)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        curr = curr + self.dropout(x)

        return curr


class SequenceMemoryFusion(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        num_layers: int = 1,
        channel_reduction: int = 8,
        dropout: float = 0.05,
        verbose: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.inner_dim = max(32, d_model // channel_reduction)
        self.verbose = verbose
        self._warned_once = False

        self.proj_in = nn.Conv2d(d_model, self.inner_dim, kernel_size=1)
        self.proj_out = nn.Conv2d(self.inner_dim, d_model, kernel_size=1)

        self.memory_layers = nn.ModuleList([
            MemoryAttentionLayer(
                d_model=self.inner_dim,
                num_heads=num_heads,
                dim_feedforward=2 * self.inner_dim,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.inner_dim)
        self.final_fuse = nn.Sequential(
            nn.Linear(self.inner_dim * 3, self.inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.inner_dim, self.inner_dim),
        )

    @staticmethod
    def _build_pos_enc(length: int, dim: int, device: torch.device, dtype: torch.dtype):
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float) * (-torch.log(torch.tensor(10000.0, device=device)) / dim))
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _prep(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.proj_in(x)
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, inner_dim]
        return x, seq, (H, W, h, w)

    def _restore(self, seq: torch.Tensor, sizes):
        H, W, h, w = sizes
        B, hw, C = seq.shape
        x = seq.reshape(B, h, w, C).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = self.proj_out(x)
        return x

    def forward(self, modalities: dict) -> torch.Tensor:
        rgb_map, rgb_seq, sizes = self._prep(modalities['rgb'])
        _, vh_seq, _ = self._prep(modalities['vh'])
        _, vv_seq, _ = self._prep(modalities['vv'])

        B, HW, C = rgb_seq.shape
        device = rgb_seq.device
        dtype = rgb_seq.dtype

        pos_hw = self._build_pos_enc(HW, C, device, dtype)           # [1, HW, C]
        pos_2hw = self._build_pos_enc(HW * 2, C, device, dtype)      # [1, 2*HW, C]

        try:
            rgb_mem = rgb_seq
            for layer in self.memory_layers:
                rgb_mem = layer(rgb_mem, rgb_mem, curr_pos=pos_hw, memory_pos=pos_hw)

            vh_mem = vh_seq
            for layer in self.memory_layers:
                vh_mem = layer(vh_mem, rgb_mem, curr_pos=pos_hw, memory_pos=pos_hw)

            combined_memory = torch.cat([rgb_mem, vh_mem], dim=1)  # [B, 2*HW, C]
            vv_mem = vv_seq
            for layer in self.memory_layers:
                vv_mem = layer(vv_mem, combined_memory, curr_pos=pos_hw, memory_pos=pos_2hw)

            final_seq = torch.cat([rgb_mem, vh_mem, vv_mem], dim=2)  # [B, HW, 3*C]
            final_seq = self.final_fuse(final_seq)
            final_seq = self.final_norm(final_seq)

            fused = self._restore(final_seq, sizes)

            if torch.isnan(fused).any() or torch.isinf(fused).any():
                if self.verbose and not self._warned_once:
                    print("⚠️ Warning: Fused features unstable, fallback to average")
                    self._warned_once = True
                fused = (modalities['rgb'] + modalities['vh'] + modalities['vv']) / 3

            return fused

        except Exception as e:
            if self.verbose and not self._warned_once:
                print(f"⚠️ Memory Attention failed: {e}, fallback to average")
                self._warned_once = True
            return (modalities['rgb'] + modalities['vh'] + modalities['vv']) / 3