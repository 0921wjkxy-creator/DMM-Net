import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeSegmentation(nn.Module):

    def __init__(self, num_classes: int, feature_dim: int, momentum: float = 0.8, device: str = "cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.register_buffer(
            "global_prototypes",
            torch.zeros(num_classes, feature_dim, dtype=torch.float32, device=device)
        )
        self._inited = False

    @torch.no_grad()
    def _update_global(self, current):
        if not self._inited:
            self.global_prototypes.copy_(current)
            self._inited = True
        else:
            self.global_prototypes.mul_(self.momentum).add_(current * (1.0 - self.momentum))

    def _batch_prototypes(self, features: torch.Tensor, labels_idx: torch.Tensor):
        """
        features: [B, C, H, W]  —— m_feat
        labels_idx: [B, H, W]   —— GT
        """
        B, C, H, W = features.shape
        labels_resized = F.interpolate(labels_idx.unsqueeze(1).float(), size=(H, W), mode="nearest").long().squeeze(1)
        feat_flat  = features.permute(0, 2, 3, 1).reshape(-1, C)   # [BHW, C]
        label_flat = labels_resized.reshape(-1)                    # [BHW]

        batch_proto = torch.zeros(self.num_classes, C, device=features.device)
        for k in range(self.num_classes):
            m = (label_flat == k)
            if m.any():
                batch_proto[k] = feat_flat[m].mean(dim=0)
            else:
                batch_proto[k] = self.global_prototypes[k].detach()
        return batch_proto

    def forward(self, m_feat: torch.Tensor, labels_idx: torch.Tensor):

        batch_proto = self._batch_prototypes(m_feat, labels_idx)
        self._update_global(batch_proto)
        return F.mse_loss(batch_proto, self.global_prototypes) 