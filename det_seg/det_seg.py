import torch, math
from torch import device, nn
from torch.nn import functional as F
from .mask_feat import MaskFeatHead
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from modules.cv.models.detection import FasterRCNN, resnet_fpn_backbone
from modules.cv.preprocess import data_mixup

class SegmentHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegmentHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)
    def forward(self, inputs, image_size):
        return F.interpolate(self.conv(inputs), image_size, mode = 'bilinear', align_corners = True)

class Rcnn_Deeplab(nn.Module):
    def __init__(self, backbone, det_classes, seg_classes, **kwargs):
        super(Rcnn_Deeplab, self).__init__()
        self.backbone, out_channels = backbone, backbone.out_channels
        self.mask_feat_head = MaskFeatHead(out_channels, 4)
        self.seg_head = SegmentHead(out_channels, seg_classes)
        self.det_head = FasterRCNN(out_channels, det_classes, **kwargs)
        self.focal_loss = nn.BCELoss(reduction = 'none')
        self.seg_classes, self.mixup = seg_classes, data_mixup()

    def forward(self, images, seg_size, gt_segs = None, gt_boxes = None):
        """
        This module wraps up the retinanet model for detection and deeplabv3 for segmentation.
        Note that all inputs are results after resizing.
        Parameters:
            images: Tensor of shape (n, C, H, W).
            seg_size: Tuple (H, W), representing the size feature maps are interpolated into after
                    the last conv layer.
            gt_boxes: list[Dict[str, Tensor]], each element in this list is a dict with keys
                'boxes', 'labels', where 'boxes' is a tensor of shape (n, 4) representing
                bounding boxes coordinates for an image, the four numbers are x1, y1, x2, y2. 
                'labels' is a tensor of shape (n) representing the label that boxes have.
        Returns:
            During training, this module returns a dict with keys 'det_loss' representing the
            detection loss and 'seg_loss' representing the segmentation loss. Both of them are
            of shape (batch_size).
            During inference, this module returns a tuple of (det_preds, seg_preds).
            det_preds is a list of dicts, each of which contains three
            keys, 'boxes', 'labels', 'scores'. 'boxes' is a tensor of shape (n, 4), 'labels'
            is a tensor of shape (n), 'scores' is a tensor of shape (n). seg_preds is a
            segmentation tensor of shape (n, H, W).
        """
        if gt_segs is not None or gt_boxes is not None:
            assert gt_segs is not None and gt_boxes is not None
            images, idxs, alpha = self.mixup(images)
            dtype, device = images.dtype, images.device
            gt_boxes_mixed = []
            for box, mixed_idx in zip(gt_boxes, idxs):
                box_permuted = gt_boxes[mixed_idx]
                w1 = torch.full((len(box['labels']),), alpha, dtype = dtype, device = device)
                w2 = torch.full((len(box_permuted['labels']),), 1 - alpha, dtype = dtype, device = device)
                gt_boxes_mixed.append({'boxes': torch.cat([box['boxes'], box_permuted['boxes']]),
                                'labels': torch.cat([box['labels'], box_permuted['labels']]),
                                'weights': torch.cat([w1, w2]) * 2})
            feats = list(self.backbone(images).values())
            det_loss = self.det_head(feats, images.shape[-2:], gt_boxes_mixed)
            seg_preds = self.seg_head(self.mask_feat_head(feats[:4]), seg_size).sigmoid()
            N, H, W = gt_segs.shape
            gt_segs = torch.zeros([N, self.seg_classes + 1, H, W], dtype = torch.float32,
                    device = gt_segs.device).scatter_(1, gt_segs[:, None], 1)[:, 1:]
            pos_num = gt_segs.sum([0, 2, 3])
            pos_num[pos_num < 1e-6] = math.sqrt(H * W)
            seg_loss1 = torch.mean(self.focal_loss(seg_preds, gt_segs).sum([0, 2, 3]) / pos_num)
            seg_loss2 = torch.mean(self.focal_loss(seg_preds, gt_segs[idxs]).sum([0, 2, 3]) / pos_num)
            det_loss['seg_loss'] = seg_loss1 * alpha + seg_loss2 * (1 - alpha)
            return det_loss
        else:
            feats = list(self.backbone(images).values())
            seg_preds = self.seg_head(self.mask_feat_head(feats[:4]), seg_size).sigmoid()
            det_preds = self.det_head(feats, images.shape[-2:])
            return det_preds, seg_preds

def det_seg_resnet(backbone_name, det_classes, seg_classes, pretrained_backbone = True,
                    trainable_layers = 5, returned_layers = [1, 2, 3, 4], **kwargs):
    backbone = resnet_fpn_backbone(backbone_name, pretrained_backbone,
                                    trainable_layers = trainable_layers,
                                    returned_layers = returned_layers,
                                    extra_blocks = LastLevelMaxPool(),
                                    out_channels = 256)
    return Rcnn_Deeplab(backbone, det_classes, seg_classes, **kwargs)