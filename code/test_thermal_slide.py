import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
from PIL import Image

import datasets
import models
import utils


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
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={tuple(v.shape)}')
        else:
            print(f'  {k}: {v}')

    import os
    import platform

    if platform.system() == 'Windows':
        num_workers = min(2, os.cpu_count() or 1)
    else:
        num_workers = min(4, os.cpu_count() or 1)

    print(f'  DataLoader: num_workers={num_workers}')


    try:
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=False, num_workers=num_workers, pin_memory=False)
        test_batch = next(iter(loader))
        print(f'  ✓ 多进程DataLoader初始化成功 (num_workers={num_workers})')
    except Exception as e:
        print(f'  ⚠ 多进程失败，降级到单进程: {e}')
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=False)
        print(f'  ✓ 单进程DataLoader初始化成功')

    return loader


def test_thermal_slide(loader, model, config, save_dir=None):
    model.eval()
    eval_type = config.get('eval_type')
    class_num = config['model']['args']['num_classes']
    ignore_background = config['test_dataset']['dataset']['args'].get('ignore_bg', False)

    if eval_type == 'seg':
        metric_seg = SegmentationMetric(class_num, ignore_background)

    pbar = tqdm(total=len(loader), leave=False, desc='test')

    for batch_idx, batch in enumerate(loader):
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

        if eval_type == 'seg':
            metric_seg.addBatch(mask_index_label, gt_index_label)

        filename = batch['filename'][0] if isinstance(batch['filename'], (list, tuple)) else batch['filename']
        save_prediction_result(output_mask, mask_index_label, filename, save_dir, config)

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
    normed_confusionMatrix = conf_matrix / conf_matrix.sum(axis=0)
    fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()

    print(f'\n=== 热融滑塌识别测试结果 ===')
    print(f'Overall Accuracy: {oa:.4f}')
    print(f'Mean IoU: {mIoU:.4f}')
    print(f'Mean Precision: {mp:.4f}')
    print(f'Mean Recall: {mr:.4f}')
    print(f'Mean F1-Score: {mf1:.4f}')
    print(f'Frequency Weighted IoU: {fwIOU:.4f}')

    class_names = config['test_dataset']['dataset']['args']['classes']
    print(f'\n各类别IoU:')
    for i, (name, iou) in enumerate(zip(class_names, IoU)):
        print(f'  {name}: {iou:.4f}')

    if save_dir is not None:
        save_test_results(oa, mIoU, mp, mr, mf1, fwIOU, IoU, class_names, save_dir)

    return oa, mIoU, mp, mr, mf1, fwIOU, IoU


def save_prediction_result(output_mask, mask_index_label, filename, save_dir, config):
    import cv2
    import numpy as np
    import os
    import torch

    os.makedirs(save_dir, exist_ok=True)
    pred_mask = mask_index_label.reshape(1024, 1024)
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()

    palette = config['test_dataset']['dataset']['args']['palette']
    colored_pred = np.zeros((1024, 1024, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        colored_pred[pred_mask == i] = color

    pred_path = os.path.join(save_dir, f'pred_{filename}')
    cv2.imwrite(pred_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
    vv_dir = config['test_dataset']['dataset']['args']['vv_root_path']
    vh_dir = config['test_dataset']['dataset']['args']['vh_root_path']
    rgb_dir = config['test_dataset']['dataset']['args']['rgb_root_path']
    lab_dir = config['test_dataset']['dataset']['args']['lab_root_path']
    vv_img = cv2.imread(os.path.join(vv_dir, filename), cv2.IMREAD_GRAYSCALE)
    vh_img = cv2.imread(os.path.join(vh_dir, filename), cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.imread(os.path.join(rgb_dir, filename))
    lab_img = cv2.imread(os.path.join(lab_dir, filename))
    vv_img = cv2.resize(vv_img, (1024, 1024))
    vh_img = cv2.resize(vh_img, (1024, 1024))
    rgb_img = cv2.resize(rgb_img, (1024, 1024))
    lab_img = cv2.resize(lab_img, (1024, 1024))
    vv_img_3ch = cv2.cvtColor(vv_img, cv2.COLOR_GRAY2BGR)
    vh_img_3ch = cv2.cvtColor(vh_img, cv2.COLOR_GRAY2BGR)

    compare_img = np.concatenate([
        vv_img_3ch, vh_img_3ch, rgb_img, lab_img, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR)
    ], axis=1)
    compare_path = os.path.join(save_dir, f'compare_{filename}')
    cv2.imwrite(compare_path, compare_img)


def save_test_results(oa, mIoU, mp, mr, mf1, fwIOU, IoU, class_names, save_dir):
    result_file = os.path.join(save_dir, 'test_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write('=== 热融滑塌识别测试结果 ===\n')
        f.write(f'Overall Accuracy: {oa:.4f}\n')
        f.write(f'Mean IoU: {mIoU:.4f}\n')
        f.write(f'Mean Precision: {mp:.4f}\n')
        f.write(f'Mean Recall: {mr:.4f}\n')
        f.write(f'Mean F1-Score: {mf1:.4f}\n')
        f.write(f'Frequency Weighted IoU: {fwIOU:.4f}\n')
        f.write(f'\n各类别IoU:\n')
        for name, iou in zip(class_names, IoU):
            f.write(f'  {name}: {iou:.4f}\n')


def main(config_, model_path, save_dir=None):
    global config
    config = config_

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

        print(f"✓ 测试时随机种子设置为: {seed}")
    else:
        print("⚠ 未设置随机种子，结果可能不可复现")
    # ============================================================

    test_loader = make_data_loader(config.get('test_dataset'), tag='test')

    model = models.make(config['model']).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        # 完整的checkpoint格式
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'✓ 加载完整checkpoint: epoch={checkpoint.get("epoch", "unknown")}, '
              f'lr={checkpoint.get("learning_rate", "unknown")}, '
              f'max_val_v={checkpoint.get("max_val_v", "unknown")}')
    else:
        model.load_state_dict(checkpoint, strict=False)
        print('✓ 加载模型权重')

    model.eval()

    print(f'模型已加载: {model_path}')
    print(f'设备: {device}')

    if save_dir is not None:
        print(f'结果将保存到: {save_dir}')

    test_thermal_slide(test_loader, model, config, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sentinel1-multimodal-slide-detection.yaml')
    parser.add_argument('--model_path', required=True, help='训练好的模型路径')
    parser.add_argument('--save_dir', default='./test_results', help='测试结果保存目录')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    main(config, args.model_path, args.save_dir)