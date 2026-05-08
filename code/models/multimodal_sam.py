import torch
import torch.nn as nn
from models.mmseg.models.sam.image_encoder import ImageEncoderViT

class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x, H, W

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, feats):
        # feats: list of [B, C, H, W]
        x = torch.cat(feats, dim=1)
        return self.conv1x1(x)

class MultiModalCWSAM(nn.Module):
    def __init__(self, num_classes=2, inp_size=1024, loss='bce', loss_weight=None, embed_dim=256, patch_size=16, stride=16, fusion_out_chans=256, encoder_mode=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.inp_size = inp_size
        self.loss_type = loss
        self.loss_weight = loss_weight
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride
        self.fusion_out_chans = fusion_out_chans
        self.encoder_mode = encoder_mode or {}
        vit_args = dict(
            img_size=self.encoder_mode.get('img_size', 1024),
            patch_size=self.encoder_mode.get('patch_size', 16),
            in_chans=3,
            embed_dim=self.encoder_mode.get('embed_dim', 768),
            depth=self.encoder_mode.get('depth', 12),
            num_heads=self.encoder_mode.get('num_heads', 12),
            mlp_ratio=self.encoder_mode.get('mlp_ratio', 4.0),
            out_chans=self.encoder_mode.get('out_chans', 256),
            qkv_bias=self.encoder_mode.get('qkv_bias', True),
            use_rel_pos=self.encoder_mode.get('use_rel_pos', False),
            window_size=self.encoder_mode.get('window_size', 0),
            global_attn_indexes=tuple(self.encoder_mode.get('global_attn_indexes', [2,5,8,11]))
        )
        self.vit_encoder_rgb = ImageEncoderViT(**vit_args)
        self.vit_encoder_vv = ImageEncoderViT(**vit_args)
        self.vit_encoder_vh = ImageEncoderViT(**vit_args)
        self.fusion_module = FusionModule(vit_args['out_chans'] * 3, fusion_out_chans)
        if loss == 'bce' and num_classes == 2:
            out_ch = 1
        else:
            out_ch = num_classes
        self.head = nn.Conv2d(fusion_out_chans, out_ch, kernel_size=1)
        if loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == 'ce':
            self.criterion = nn.CrossEntropyLoss(weight=None if loss_weight is None else torch.tensor(loss_weight))
        else:
            raise NotImplementedError(f"Unknown loss: {loss}")
        self.optimizer = None
        self.loss_G = None
        self.input = None
        self.gt_mask = None

    def set_input(self, x, gt_mask):
        self.input = x
        self.gt_mask = gt_mask

    def forward(self, x):
        print("Start forward")
        rgb = x['rgb']
        vv = x['vv'].repeat(1, 3, 1, 1)
        vh = x['vh'].repeat(1, 3, 1, 1)
        print("Before vit_encoder_rgb")
        rgb_encoded = self.vit_encoder_rgb(rgb)
        print("Before vit_encoder_vv")
        vv_encoded = self.vit_encoder_vv(vv)
        print("Before vit_encoder_vh")
        vh_encoded = self.vit_encoder_vh(vh)
        print("After all vit encoders")
        fused_feat = self.fusion_module([rgb_encoded, vv_encoded, vh_encoded])
        out = self.head(fused_feat)
        print("End forward")
        return out

    def optimize_parameters(self):
        self.train()
        self.optimizer.zero_grad()
        output = self.forward(self.input)
        gt = self.gt_mask
        if self.loss_type == 'bce':
            if gt.shape[1] == 2:
                gt = gt[:, 0:1, :, :]
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            if gt.shape[1] != 1:
                gt = gt[:, 0:1, ...]
            gt = gt.float()
            if output.shape != gt.shape:
                gt = torch.nn.functional.interpolate(gt, size=output.shape[-2:], mode='nearest')
            loss = self.criterion(output, gt)
        else:
            if gt.dim() == 4 and gt.shape[1] == 1:
                gt = gt[:, 0, :, :]
            if output.shape[-2:] != gt.shape[-2:]:
                gt = torch.nn.functional.interpolate(gt.unsqueeze(1).float(), size=output.shape[-2:], mode='nearest').long().squeeze(1)
            loss = self.criterion(output, gt)
        loss.backward()
        self.optimizer.step()
        self.loss_G = loss

    def infer(self, batch):
        self.eval()
        with torch.no_grad():
            output = self.forward({'vv': batch['vv'], 'vh': batch['vh'], 'rgb': batch['rgb']})
        return output