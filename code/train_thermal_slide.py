import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch
import numpy as np
import math
import random

import datasets
import models
import models.mmseg.models.sam.semantic_prototype_memory as spmm
import utils
from statistics import mean
import os


class WarmupCosineAnnealingLR:

    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.last_epoch += 1

        if self.last_epoch < self.warmup_epochs:
            lr = self.base_lrs[0] * (self.last_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'eta_min': self.eta_min,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']
        self.eta_min = state_dict['eta_min']
        self.base_lrs = state_dict['base_lrs']


class WarmupCosineAnnealingWarmRestarts:

    def __init__(self, optimizer, warmup_epochs, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.T_cur = 0
        self.T_i = T_0

    def step(self):
        self.last_epoch += 1

        if self.last_epoch < self.warmup_epochs:
            lr = self.base_lrs[0] * (self.last_epoch + 1) / self.warmup_epochs
        else:
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i *= self.T_mult

            progress = self.T_cur / self.T_i
            lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
            self.T_cur += 1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {
            'last_epoch': self.last_epoch,
            'warmup_epochs': self.warmup_epochs,
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'eta_min': self.eta_min,
            'base_lrs': self.base_lrs,
            'T_cur': self.T_cur,
            'T_i': self.T_i
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.T_0 = state_dict['T_0']
        self.T_mult = state_dict['T_mult']
        self.eta_min = state_dict['eta_min']
        self.base_lrs = state_dict['base_lrs']
        self.T_cur = state_dict['T_cur']
        self.T_i = state_dict['T_i']


def onehot_to_index_label(onehot):
    return onehot.argmax(dim=0)


class SegmentationMetric:
    def __init__(self, num_classes, ignore_bg=False):
        self.num_classes = num_classes
        self.ignore_bg = ignore_bg
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def addBatch(self, pred, gt):
        if len(pred.shape) == 1:
            pred = pred.reshape(-1)
        if len(gt.shape) == 1:
            gt = gt.reshape(-1)

        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.cpu().numpy()

        mask = (gt >= 0) & (gt < self.num_classes)
        label = self.num_classes * gt[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion_matrix

    def overallAccuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix)
        union = np.where(union == 0, 1, union)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU, IoU

    def precision(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=0)
        IoU = intersection / union
        return IoU

    def recall(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1)
        IoU = intersection / union
        return IoU

    def confusionMatrix(self):
        return self.confusion_matrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    print(f'{tag} dataset: size={len(dataset)}')
    for k, v in dataset[0].items():
        if isinstance(v, str):
            print(f'  {k}: {v}')
        else:
            print(f'  {k}: shape={tuple(v.shape)}')

    import os
    import platform

    if platform.system() == 'Windows':
        num_workers = min(2, os.cpu_count() or 1)
    else:
        num_workers = min(4, os.cpu_count() or 1)

    print(f'  DataLoader: num_workers={num_workers}')

    try:
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=(tag == 'train'), num_workers=num_workers, pin_memory=False)
        test_batch = next(iter(loader))
        print(f'  ✓ 多进程DataLoader初始化成功 (num_workers={num_workers})')
    except Exception as e:
        print(f'  ⚠ 多进程失败，降级到单进程: {e}')
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=(tag == 'train'), num_workers=0, pin_memory=False)
        print(f'  ✓ 单进程DataLoader初始化成功')

    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_thermal_slide(loader, model, config):
    model.eval()
    eval_type = config.get('eval_type')
    class_num = config['model']['args']['num_classes']
    ignore_background = config['val_dataset']['dataset']['args'].get('ignore_bg', False)

    if eval_type == 'seg':
        metric_seg = SegmentationMetric(class_num, ignore_background)

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(total=len(loader), leave=False, desc='val')

    for i, batch in enumerate(loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            output_masks = model.infer(batch)
            pred = torch.sigmoid(output_masks)

        output_mask = pred[0]
        mask_index_label = onehot_to_index_label(output_mask).flatten()
        gt_mask = batch['gt'][0]
        gt_index_label = onehot_to_index_label(gt_mask).flatten()

        if i < 3:
            unique, counts = torch.unique(mask_index_label, return_counts=True)
            print("预测类别分布:", dict(zip(unique.tolist(), counts.tolist())))
            unique_gt, counts_gt = torch.unique(gt_index_label, return_counts=True)
            print("GT类别分布:", dict(zip(unique_gt.tolist(), counts_gt.tolist())))

        if eval_type == 'seg':
            metric_seg.addBatch(mask_index_label, gt_index_label)

        result1, result2, result3, result4 = utils.calc_cod(pred, batch['gt'])

        val_metric1.add(result1.item(), batch['vv'].shape[0])
        val_metric2.add(result2.item(), batch['vv'].shape[0])
        val_metric3.add(result3.item(), batch['vv'].shape[0])
        val_metric4.add(result4.item(), batch['vv'].shape[0])

        pbar.update(1)

    pbar.close()

    oa = metric_seg.overallAccuracy()
    mIoU, IoU = metric_seg.meanIntersectionOverUnion()
    p = metric_seg.precision()
    mp = np.nanmean(p)
    r = metric_seg.recall()
    mr = np.nanmean(r)
    p = np.array(p)
    r = np.array(r)
    f1 = (2 * p * r) / (p + r)
    mf1 = np.nanmean(f1)
    conf_matrix = metric_seg.confusionMatrix()
    col_sums = conf_matrix.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    normed_confusionMatrix = conf_matrix / col_sums
    fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()

    print(f'\n=== 热融滑塌识别评估结果 ===')
    print(f'Overall Accuracy: {oa:.4f}')
    print(f'Mean IoU: {mIoU:.4f}')
    print(f'Mean Precision: {mp:.4f}')
    print(f'Mean Recall: {mr:.4f}')
    print(f'Mean F1-Score: {mf1:.4f}')
    print(f'Frequency Weighted IoU: {fwIOU:.4f}')

    class_names = config['train_dataset']['dataset']['args']['classes']
    print(f'\n各类别IoU:')
    for i, (name, iou) in enumerate(zip(class_names, IoU)):
        print(f'  {name}: {iou:.4f}')

    thermal_slide_iou = None
    if eval_type == 'seg':
        class_names = config['train_dataset']['dataset']['args']['classes']
        if 'slide' in class_names:
            thermal_slide_idx = class_names.index('slide')
            thermal_slide_iou = IoU[thermal_slide_idx]
        elif 'thermal_slide' in class_names:
            thermal_slide_idx = class_names.index('thermal_slide')
            thermal_slide_iou = IoU[thermal_slide_idx]

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), \
        'sm', 'em', 'wfm', 'mae', thermal_slide_iou, normed_confusionMatrix


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
        resume_model_path = os.path.join(config.get('work_dir'), 'model_epoch_' + str(config.get('resume')) + '.pth')
        resume_checkpoint = torch.load(resume_model_path)
        model.load_state_dict(resume_checkpoint, strict=False)
    else:
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1

    max_epoch = config.get('epoch_max')

    scheduler_type = config.get('scheduler', {}).get('type', 'cosine')  # 默认使用余弦退火
    scheduler_config = config.get('scheduler', {})

    if scheduler_type == 'warmup_cosine':
        warmup_epochs = scheduler_config.get('warmup_epochs', max(1, int(max_epoch * 0.05)))  # 默认5%的epoch用于warmup
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer, warmup_epochs, max_epoch,
            eta_min=config.get('lr_min')
        )
        print(f"✓ 使用带warmup的余弦退火调度器 (warmup: {warmup_epochs} epochs)")

    elif scheduler_type == 'warmup_cosine_restart':
        warmup_epochs = scheduler_config.get('warmup_epochs', max(1, int(max_epoch * 0.05)))
        T_0 = scheduler_config.get('T_0', max_epoch // 3)
        T_mult = scheduler_config.get('T_mult', 2)
        lr_scheduler = WarmupCosineAnnealingWarmRestarts(
            optimizer, warmup_epochs, T_0, T_mult,
            eta_min=config.get('lr_min')
        )
        print(f"✓ 使用带warmup的余弦退火带重启调度器 (warmup: {warmup_epochs}, T_0: {T_0}, T_mult: {T_mult})")

    elif scheduler_type == 'multistep':
        # 兼容 MultiStepLR（来自旧配置）
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = config.get('multi_step_lr', {}).get('milestones', [])
        gamma = config.get('multi_step_lr', {}).get('gamma', 0.1)
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        print(f"✓ 使用MultiStepLR调度器 (milestones={milestones}, gamma={gamma})")
    else:
        # 默认：标准余弦退火
        lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
        print(f"✓ 使用标准余弦退火调度器")

    print(f'model: #params={utils.compute_num_params(model, text=True)}')

    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, prototypeseg=None):
    model.train()

    accumulation_steps = config.get('gradient_accumulation', {}).get('steps', 1)
    gradient_clip_norm = config.get('gradient_clipping', {}).get('max_norm', None)

    pbar = tqdm(total=len(train_loader), leave=False, desc='train')

    loss_list = []
    proto_loss_list = []
    gradient_norms = []

    for batch_idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        gt = batch['gt']
        model.set_input(batch, gt)


        model.optimize_parameters()

        main_loss = model.loss_G.item()
        loss_list.append(main_loss)

        if prototypeseg is not None:

            m_feat = model.m_feat_semantic  # [B, proto_dim, Hf, Wf]
            gt_onehot = gt  # [B, 2, H, W]
            gt_labels = torch.argmax(gt_onehot, dim=1)  # [B, H, W]

            proto_loss = prototypeseg(m_feat, gt_labels)

            Hf, Wf = m_feat.shape[-2:]
            proto_lambda = config.get('model', {}).get('args', {}).get('encoder_mode', {}).get('semantic_prototype',
                                                                                               {}).get('lambda', 0.01)
            proto_loss_weighted = proto_lambda * proto_loss * (Hf * Wf)

            proto_loss_list.append(proto_loss_weighted.item())

            total_loss = main_loss + proto_loss_weighted
            model.loss_G = torch.tensor(total_loss, device=device)

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        total_norm = 0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            gradient_norms.append(total_norm)

        pbar.update(1)

    pbar.close()

    loss = mean(loss_list)
    avg_grad_norm = mean(gradient_norms) if gradient_norms else 0.0

    if prototypeseg is not None and proto_loss_list:
        proto_loss_avg = mean(proto_loss_list)
        print(f"  - 主损失: {loss:.4f}, 原型损失: {proto_loss_avg:.4f}, 梯度范数: {avg_grad_norm:.4f}")
    else:
        print(f"  - 主损失: {loss:.4f}, 梯度范数: {avg_grad_norm:.4f}")

    return loss, avg_grad_norm


def save(config, model, save_path, name):
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f'model_{name}.pth'))


def save_checkpoint(model, optimizer, lr_scheduler, epoch, save_path, name, max_val_v=None, train_loss=None,
                    grad_norm=None):
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    if max_val_v is not None:
        checkpoint['max_val_v'] = max_val_v
    if train_loss is not None:
        checkpoint['train_loss'] = train_loss
    if grad_norm is not None:
        checkpoint['grad_norm'] = grad_norm
    torch.save(checkpoint, os.path.join(save_path, f'checkpoint_{name}.pth'))


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    seed = config.get('seed', None)
    if seed is not None:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"✓ 随机种子设置为: {seed}")
        print(f"✓ 已启用确定性训练模式")
    else:
        print("⚠ 未设置随机种子，训练结果可能不可复现")
    checkpoint_path = os.path.join(save_path, 'checkpoint_last.pth')
    resume_epoch = 1
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    if os.path.exists(checkpoint_path):
        print(f"检测到断点文件，正在恢复训练: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        resume_epoch = checkpoint['epoch']
        max_val_v = checkpoint.get('max_val_v', -1e18 if config['eval_type'] != 'ber' else 1e8)

        best_checkpoint_path = os.path.join(save_path, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint_path):
            try:
                best_ckpt = torch.load(best_checkpoint_path, map_location='cpu')
                if 'max_val_v' in best_ckpt:
                    print(f"✓ 从最佳模型checkpoint恢复 max_val_v={best_ckpt['max_val_v']:.4f}")
                    max_val_v = best_ckpt['max_val_v']
            except Exception as e:
                print(f"⚠ 无法从最佳模型checkpoint读取 max_val_v: {e}")

        print(f"✓ 断点文件加载成功，从第 {resume_epoch} 个epoch重新开始训练")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']}")
        if 'train_loss' in checkpoint:
            print(f"上次训练损失: {checkpoint['train_loss']:.4f}")
        if 'grad_norm' in checkpoint:
            print(f"上次梯度范数: {checkpoint['grad_norm']:.4f}")
    else:
        resume_epoch = 1
        max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    # ============================================================

    train_loader, val_loader = make_data_loaders()

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model.optimizer = optimizer

    if resume_epoch > 1:
        print("✓ 使用断点文件中的学习率调度器状态")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']}")
    else:
        print("✓ 从头开始训练，使用默认学习率")

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in sam_checkpoint.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"成功加载 {len(pretrained_dict)}/{len(sam_checkpoint)} 个预训练参数")

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
        if "image_encoder" in name and "Adapter" in name:
            para.requires_grad_(True)

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
    log('model_grad_params:' + str(model_grad_params) + '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    timer = utils.Timer()

    print(f'\n=== 开始热融滑塌识别模型训练 ===')
    print(f'训练轮数: {epoch_max}')
    print(f'学习率: {optimizer.param_groups[0]["lr"]}')
    print(f'设备: {device}')

    scheduler_type = config.get('scheduler', {}).get('type', 'cosine')
    accumulation_steps = config.get('gradient_accumulation', {}).get('steps', 1)
    gradient_clip_norm = config.get('gradient_clipping', {}).get('max_norm', None)

    print(f'\n=== 训练优化策略配置 ===')
    print(f'学习率调度器: {scheduler_type}')
    if accumulation_steps > 1:
        print(f'梯度累积步数: {accumulation_steps}')
    if gradient_clip_norm is not None:
        print(f'梯度裁剪阈值: {gradient_clip_norm}')
    print('=' * 40)

    if (config.get('model', {}).get('args', {}).get('encoder_mode', {}).get('semantic_prototype', {}).get('enabled',
                                                                                                          False)):
        num_classes = config['model']['args']['num_classes']
        proto_dim = config['model']['args']['encoder_mode']['semantic_prototype'].get('feature_dim', 32)
        proto_m = config['model']['args']['encoder_mode']['semantic_prototype'].get('momentum', 0.8)
        proto_lambda = 1.0
        prototypeseg = spmm.PrototypeSegmentation(
            num_classes=num_classes,
            feature_dim=proto_dim,
            momentum=proto_m,
            device=device
        ).to(device)

        proto_lambda = config.get('model', {}).get('args', {}).get('encoder_mode', {}).get('semantic_prototype',
                                                                                           {}).get('lambda', 0.01)
        print(f"✓ SPMM 初始化完成:")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 原型特征维度: {proto_dim}")
        print(f"  - 动量: {proto_m}")
        print(f"  - 损失系数λ: {proto_lambda}")
    else:
        prototypeseg = None
        print("ℹ SPMM 未启用")

    # ======================== 训练循环 ========================
    for epoch in range(resume_epoch, epoch_max + 1):
        t_epoch_start = timer.t()
        train_loss_G, avg_grad_norm = train(train_loader, model, prototypeseg)
        lr_scheduler.step()

        log_info = [f'\n ############################ epoch {epoch}/{epoch_max} ############################']

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        writer.add_scalar('Gradient_Norm', avg_grad_norm, epoch)

        log_info.append(f'train G: loss={train_loss_G:.4f}, lr={current_lr:.6f}, grad_norm={avg_grad_norm:.4f}')
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        save_checkpoint(model, optimizer, lr_scheduler, epoch, save_path, 'last', max_val_v, train_loss_G,
                        avg_grad_norm)

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            with torch.no_grad():
                result1, result2, result3, result4, metric1, metric2, metric3, metric4, thermal_slide_iou, normed_confusionMatrix = eval_thermal_slide(
                    val_loader, model, config)

            if config['eval_type'] == 'seg':
                if thermal_slide_iou is not None:
                    if thermal_slide_iou > max_val_v:
                        max_val_v = thermal_slide_iou
                        save_checkpoint(model, optimizer, lr_scheduler, epoch, save_path, 'best', max_val_v,
                                        train_loss_G, avg_grad_norm)
                        print(f'保存最佳模型checkpoint，热融滑塌IoU: {thermal_slide_iou:.4f}')
            else:
                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save_checkpoint(model, optimizer, lr_scheduler, epoch, save_path, 'best', max_val_v,
                                        train_loss_G, avg_grad_norm)
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save_checkpoint(model, optimizer, lr_scheduler, epoch, save_path, 'best', max_val_v,
                                        train_loss_G, avg_grad_norm)

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1) * 100
            log_info.append(f'val G: loss={result1:.4f}, epoch time={t:.1f}s, progress={prog:.1f}%')

            print('\n'.join(log_info))
            timer.reset()
        else:
            print('\n'.join(log_info))
            timer.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='C:\\Users\\Student\\Desktop\\cwsam\\configs\\sentinel1-multimodal-slide-detection.yaml')
    parser.add_argument('--work_dir', default='./work_dir_thermal_slide')
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    save_path = args.work_dir
    main(config, save_path, args) 