import math, torch
from torch import nn, Tensor
from typing import Dict, List, Tuple
from .utils import *
from ...ops import FocalLoss, SmoothL1Loss
from torchvision.ops import boxes as box_ops

class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.cls_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.reg_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
                    anchors: List[Tensor], matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'classification': self.cls_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.reg_head.compute_loss(targets, head_outputs, anchors, matched_idxs)
        }

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            'cls_logits': self.cls_head(x),
            'bbox_regression': self.reg_head(x)
        }

class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, prior_prob = 0.01):
        super().__init__()
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, 3, stride = 1, padding = 1)
        torch.nn.init.normal_(self.cls_logits.weight, std = 0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_prob) / prior_prob))
        self.num_classes, self.num_anchors = num_classes, num_anchors
        self.BETWEEN_THRESHOLDS, self.focal_loss = -2, FocalLoss(reduction = 'none')

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
        matched_idxs: List[Tensor]) -> Tensor:
        losses, cls_logits = [], head_outputs['cls_logits']
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in\
            zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()
            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_weights = torch.ones([cls_logits_per_image.shape[0]], dtype = torch.float32,
                                        device = cls_logits_per_image.device)
            weights = targets_per_image.get('weights')
            matched_foreground = matched_idxs_per_image[foreground_idxs_per_image]
            if weights is not None:
                gt_weights[foreground_idxs_per_image] = weights[matched_foreground]
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image['labels'][matched_foreground]
            ] = 1.0
            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS
            # compute the classification loss
            loss = self.focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image]
            )
            loss = torch.sum(loss.sum(-1) * gt_weights[valid_idxs_per_image]) / max(1, num_foreground)
            losses.append(loss)
        return sum(losses) / max(1, len(targets))

    def forward(self, x: List[Tensor]) -> Tensor:
        all_cls_logits = []
        for features in x:
            cls_logits = self.cls_logits(self.conv(features))
            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)
            all_cls_logits.append(cls_logits)
        return torch.cat(all_cls_logits, 1)

class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    __annotations__ = {'box_coder': BoxCoder}
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding = 1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)
        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, 3, stride = 1, padding = 1)
        torch.nn.init.normal_(self.bbox_reg.weight, std = 0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)
        self.box_coder, self.l1_loss = BoxCoder((1.0, 1.0, 1.0, 1.0)), SmoothL1Loss(reduction = 'none')

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
        anchors: List[Tensor], matched_idxs: List[Tensor]) -> Tensor:
        losses, bbox_regression = [], head_outputs['bbox_regression']
        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()
            # select only the foreground boxes
            weights = targets_per_image.get('weights')
            if weights is None:
                weights = torch.ones([len(targets_per_image['labels'])], dtype = torch.float32,
                                        device = anchors_per_image.device)
            matched_foreground = matched_idxs_per_image[foreground_idxs_per_image]
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_foreground]
            matched_gt_weights_per_image = weights[matched_foreground]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            # compute the loss
            loss = self.l1_loss(bbox_regression_per_image, target_regression).sum(-1)
            loss = torch.sum(loss * matched_gt_weights_per_image) / max(1, num_foreground)
            losses.append(loss)
        return sum(losses) / max(1, len(targets))

    def forward(self, x: List[Tensor]) -> Tensor:
        all_bbox_regression = []
        for features in x:
            bbox_regression = self.bbox_reg(self.conv(features))
            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)
            all_bbox_regression.append(bbox_regression)
        return torch.cat(all_bbox_regression, 1)

class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        out_channels (int): backbone out_channels.
        num_classes (int): number of output classes of the model (excluding the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (List[float]): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.
    """
    __annotations__ = {'box_coder': BoxCoder, 'proposal_matcher': Matcher}
    def __init__(self, out_channels, num_classes, anchor_generator = None, head = None,
                 score_thresh = 0.05, nms_thresh = 0.5, detections_per_img = 300,
                 fg_iou_thresh = 0.5, bg_iou_thresh = 0.4, topk_candidates = 1000):
        super().__init__()
        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))
        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                                for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator
        if head is None:
            head = RetinaNetHead(out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, True)
        self.box_coder = BoxCoder((1.0, 1.0, 1.0, 1.0))
        if isinstance(score_thresh, float) or isinstance(score_thresh, int):
            score_thresh = [score_thresh] * num_classes
        self.score_thresh, self.nms_thresh = score_thresh, nms_thresh
        self.detections_per_img, self.topk_candidates = detections_per_img, topk_candidates

    def compute_loss(self, targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor], anchors: List[Tensor]) -> Dict[str, Tensor]:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype = torch.int64,
                                               device = anchors_per_image.device))
                continue
            match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))
        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs: Dict[str, List[Tensor]],
        anchors: List[List[Tensor]], image_shape: Tuple[int, int]) -> List[Dict[str, Tensor]]:
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits, box_regression = head_outputs['cls_logits'], head_outputs['bbox_regression']
        num_images, detections = len(anchors), []
        score_thresh = torch.as_tensor(self.score_thresh, dtype = torch.float32,
                                        device = class_logits[0].device)
        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image = anchors[index]
            image_boxes, image_scores, image_labels = [], [], []
            for box_regression_per_level, logits_per_level, anchors_per_level in \
                    zip(box_regression_per_image, logits_per_image, anchors_per_image):
                num_classes = logits_per_level.shape[-1]
                # remove low scoring boxes
                scores_per_level = logits_per_level.sigmoid()
                keep_idxs = scores_per_level > score_thresh
                scores_per_level, keep_idxs = scores_per_level.flatten(), keep_idxs.flatten()
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]
                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]
                anchor_idxs = topk_idxs // num_classes
                labels_per_level = topk_idxs % num_classes
                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)
                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)
            image_boxes = torch.cat(image_boxes)
            image_scores = torch.cat(image_scores)
            image_labels = torch.cat(image_labels)
            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            detections.append({
                'boxes': image_boxes[keep],
                'scores': image_scores[keep],
                'labels': image_labels[keep],
            })
        return detections

    def forward(self, features: List[Tensor], image_size: Tuple[int, int], targets = None):
        """
        Args:
            features (list[Tensor]): features output by the fpn network.
            image_size (Tuple[int, int]): image sizes before features are extracted.
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if targets is not None:
            check_valid_targets(targets)
        if isinstance(features, Tensor):
            features = [features]
        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)
        # create the set of anchors
        anchors = [self.anchor_generator(image_size, [feature.shape[-2:] for feature in features],
                                        features[0].device)] * features[0].shape[0]
        if targets is not None:
            detections = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW, HWA = sum(num_anchors_per_level), head_outputs['cls_logits'].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]
            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, image_size)
        return detections