from torchvision.ops import box_iou
from modules.cv.ops.masks import mask_iou
import torch

def metric4box(pred, target, class_list):
    pred_len, tar_len = len(pred['labels']), len(target['labels'])
    num_classes = len(class_list)
    if (not pred_len) and (not tar_len):
        TP, FP, FN = [0] * num_classes, [0] * num_classes, [0] * num_classes
    elif not pred_len:
        TP, FP = [0] * num_classes, [0] * num_classes
        FN = [torch.sum(target['labels'] == i).item() for i in class_list]
    elif not tar_len:
        TP, FN = [0] * num_classes, [0] * num_classes
        FP = [torch.sum(pred['labels'] == i).item() for i in class_list]
    else:
        iou_box = box_iou(pred['boxes'], target['boxes'])
        pred_label = pred['labels'][:, None].expand(-1, tar_len)
        tar_label = target['labels'][None].expand(pred_len, -1)
        label_ac = (pred_label == tar_label).long()
        box_ac = (iou_box > 0.5).long() * label_ac
        pred_box_ac = torch.zeros([pred_len], dtype = box_ac.dtype, device = box_ac.device)
        tar_box_ac = torch.zeros([tar_len], dtype = box_ac.dtype, device = box_ac.device)
        for i in range(tar_len):
            ac = torch.argsort(box_ac[:, i])
            for j in range(pred_len):
                if box_ac[ac[j], i] and (not pred_box_ac[ac[j]]) and (not tar_box_ac[i]):
                    pred_box_ac[ac[j]] = 1
                    tar_box_ac[i] = 1
        TP, FP, FN = [], [], []
        for i in class_list:
            pred_idxs, target_idxs = pred['labels'] == i, target['labels'] == i
            pred_ac, target_ac = pred_box_ac[pred_idxs], tar_box_ac[target_idxs]
            tp, fn = pred_ac.sum(), target_idxs.sum() - target_ac.sum()
            fp = pred_idxs.sum() - tp
            TP.append(tp.item())
            FP.append(fp.item())
            FN.append(fn.item())
    return TP, FP, FN

def metric4mask(pred, target, class_list):
    pred_len, tar_len = len(pred['labels']), len(target['labels'])
    num_classes = len(class_list)
    if (not pred_len) and (not tar_len):
        TP, FP, FN = [0] * num_classes, [0] * num_classes, [0] * num_classes
    elif not pred_len:
        TP, FP = [0] * num_classes, [0] * num_classes
        FN = [torch.sum(target['labels'] == i).item() for i in class_list]
    elif not tar_len:
        TP, FN = [0] * num_classes, [0] * num_classes
        FP = [torch.sum(pred['labels'] == i).item() for i in class_list]
    else:
        iou_mask = mask_iou(pred['masks'], target['masks'])
        pred_label = pred['labels'][:, None].expand(-1, tar_len)
        tar_label = target['labels'][None].expand(pred_len, -1)
        label_ac = (pred_label == tar_label).long()
        mask_ac = (iou_mask > 0.5).long() * label_ac
        pred_mask_ac = torch.zeros([pred_len], dtype = mask_ac.dtype, device = mask_ac.device)
        tar_mask_ac = torch.zeros([tar_len], dtype = mask_ac.dtype, device = mask_ac.device)
        for i in range(tar_len):
            ac = torch.argsort(mask_ac[:, i])
            for j in range(pred_len):
                if mask_ac[ac[j], i] and (not pred_mask_ac[ac[j]]) and (not tar_mask_ac[i]):
                    pred_mask_ac[ac[j]] = 1
                    tar_mask_ac[i] = 1
        TP, FP, FN = [], [], []
        for i in class_list:
            pred_idxs, target_idxs = pred['labels'] == i, target['labels'] == i
            pred_ac, target_ac = pred_mask_ac[pred_idxs], tar_mask_ac[target_idxs]
            tp, fn = pred_ac.sum(), target_idxs.sum() - target_ac.sum()
            fp = pred_idxs.sum() - tp
            TP.append(tp.item())
            FP.append(fp.item())
            FN.append(fn.item())
    return TP, FP, FN