import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1, 2, 0)
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    x = x.permute(2, 0, 1)
    return x


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''

    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None, num_classes=None, loss_weight=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False

        self.loss_mode = loss

        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            # self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            # pos_weight = torch.tensor([1.5, 1, 0.5, 1.9, 0.1], dtype=torch.float)
            if loss_weight is not None:
                weight = torch.tensor(loss_weight, dtype=torch.float)
                self.criterionBCE = torch.nn.CrossEntropyLoss(weight=weight)
            else:
                self.criterionBCE = torch.nn.CrossEntropyLoss()

            self.criterionIOU = IOU()

        # elif self.loss_mode == 'iou_ce':
        #     self.criterionBCE =  torch.nn.CrossEntropyLoss()
        #     self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])


        self.fusion_conv = nn.Conv2d(encoder_mode['out_chans'] * 3, encoder_mode['out_chans'], kernel_size=1)
        self.modality_configs = encoder_mode.get('modalities', {})
        if self.modality_configs:
            print("✓ 模态类型设置已启用:")
            for modality, config in self.modality_configs.items():
                print(f"  - {modality}: {config['input_type']}")
        else:
            print("ℹ 使用全局模态设置")

        mem_cfg = encoder_mode.get('memory_attention', {})
        self.use_memory_attention = bool(mem_cfg.get('enabled', False))
        if self.use_memory_attention:
            from .mmseg.models.sam.memory_fusion_seq import SequenceMemoryFusion
            self.memory_attention = SequenceMemoryFusion(
                d_model=encoder_mode['out_chans'],
                num_heads=mem_cfg.get('num_heads', 2),
                num_layers=mem_cfg.get('num_layers', 1),
                channel_reduction=mem_cfg.get('channel_reduction', 8),
                dropout=mem_cfg.get('dropout', 0.05),
                verbose=False,
            )
            print("✓ Memory Attention(轻量版)已启用")

        proto_cfg = encoder_mode.get('semantic_prototype', {})
        self.use_semantic_prototype = bool(proto_cfg.get('enabled', False))
        if self.use_semantic_prototype:
            self.proto_feature_dim = proto_cfg.get('feature_dim', 32)
            self.mfeat_proj = nn.Conv2d(encoder_mode['out_chans'], self.proto_feature_dim, kernel_size=1)
            print("✓ Semantic Prototype Memory Module (SPMM) 配置已启用")
            print(f"  - 原型特征维度: {self.proto_feature_dim}")

    def set_input(self, batch, gt_mask):
        self.input = batch
        self.gt_mask = gt_mask.to(self.device)

    def _encode_modality(self, x, modality_name):

        if modality_name in self.modality_configs:
            config = self.modality_configs[modality_name]
            input_type = config.get('input_type', 'fft')

            if input_type == 'fft':
                freq_nums = config.get('freq_nums', 0.25)
                if not hasattr(self, '_printed_modality_info'):
                    print(f"✓ {modality_name} 使用FFT低频增强 (freq_nums={freq_nums})")
                return self._apply_fft_enhancement(x, freq_nums)
            elif input_type == 'rgb':
                if not hasattr(self, '_printed_modality_info'):
                    print(f"✓ {modality_name} 跳过FFT，直接处理")
                return self.image_encoder(x)
            else:
                if not hasattr(self, '_printed_modality_info'):
                    print(f"ℹ {modality_name} 使用默认处理")
                return self.image_encoder(x)
        else:
            # 使用全局设置
            return self.image_encoder(x)

    def _apply_fft_enhancement(self, x, freq_nums):
        if not hasattr(self, '_printed_modality_info'):
            print(f"✓ 应用FFT低频增强 (freq_nums={freq_nums})")
        return self.image_encoder(x)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self):
        bs = 1
        vv = self.input['vv'].to(self.device)
        vh = self.input['vh'].to(self.device)
        rgb = self.input['rgb'].to(self.device)
        if hasattr(self, 'modality_configs') and self.modality_configs:
            fe_vv = self._encode_modality(vv, 'vv')
            fe_vh = self._encode_modality(vh, 'vh')
            fe_rgb = self._encode_modality(rgb, 'rgb')
            if not hasattr(self, '_printed_modality_info'):
                self._printed_modality_info = True
        else:
            fe_vv = self.image_encoder(vv)
            fe_vh = self.image_encoder(vh)
            fe_rgb = self.image_encoder(rgb)

        if hasattr(self, 'use_semantic_prototype') and self.use_semantic_prototype:
            m_feat_raw = (fe_rgb + fe_vh + fe_vv) / 3.0
            m_feat = self.mfeat_proj(m_feat_raw)  # [B, proto_feature_dim, H', W']
            self.m_feat_semantic = m_feat

            if not hasattr(self, '_printed_mfeat_info'):
                print(f"✓ m_feat 产出: {m_feat.shape}")
                self._printed_mfeat_info = True

        if not hasattr(self, '_printed_features'):
            if hasattr(self, 'use_memory_attention') and self.use_memory_attention:
                features = self.memory_attention({'rgb': fe_rgb, 'vh': fe_vh, 'vv': fe_vv})
            else:
                features = torch.cat([fe_vv, fe_vh, fe_rgb], dim=1)
                features = self.fusion_conv(features)
            self._printed_features = True
        else:
            if hasattr(self, 'use_memory_attention') and self.use_memory_attention:
                features = self.memory_attention({'rgb': fe_rgb, 'vh': fe_vh, 'vv': fe_vv})
            else:
                features = torch.cat([fe_vv, fe_vh, fe_rgb], dim=1)
                features = self.fusion_conv(features)
        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks

    def infer(self, batch):
        bs = 1
        vv = batch['vv'].to(self.device)
        vh = batch['vh'].to(self.device)
        rgb = batch['rgb'].to(self.device)

        if hasattr(self, 'modality_configs') and self.modality_configs:
            fe_vv = self._encode_modality(vv, 'vv')
            fe_vh = self._encode_modality(vh, 'vh')
            fe_rgb = self._encode_modality(rgb, 'rgb')
            if not hasattr(self, '_printed_modality_info'):
                self._printed_modality_info = True
        else:
            fe_vv = self.image_encoder(vv)
            fe_vh = self.image_encoder(vh)
            fe_rgb = self.image_encoder(rgb)
        if hasattr(self, 'use_memory_attention') and self.use_memory_attention:
            features = self.memory_attention({'rgb': fe_rgb, 'vh': fe_vh, 'vv': fe_vv})
        else:
            features = torch.cat([fe_vv, fe_vh, fe_rgb], dim=1)
            features = self.fusion_conv(features)  # 降维到256
        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = masks[0]
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        if self.loss_mode == 'iou':
            gt_labels = torch.argmax(self.gt_mask, dim=1)  # [B, H, W]

            ce_loss = self.criterionBCE(self.pred_mask, gt_labels)
            gt_binary = self.gt_mask[:, 0:1, :, :]
            iou_loss = self.criterionIOU(torch.sigmoid(self.pred_mask), gt_binary)

            self.loss_G = ce_loss + iou_loss
        else:
            self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
