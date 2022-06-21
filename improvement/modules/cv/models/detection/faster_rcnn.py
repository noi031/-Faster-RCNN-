import torchvision, torch, math
from torch import device, nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign, boxes as box_ops
from .utils import *
from typing import List, Dict, Tuple
from ...ops import SmoothL1Loss, FocalLoss

@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob: Tensor, orig_pre_nms_top_n: int) -> Tuple[int, int]:
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))
    return num_anchors, pre_nms_top_n

class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    def __init__(self, in_channels: int, num_anchors: int):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding = 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1, stride = 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1, stride = 1)
        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits, bbox_reg = [], []
        for feature in x:
            t = self.conv(feature).relu()
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

def permute_and_flatten(layer: Tensor, N, A, C, H, W) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(box_cls: List[Tensor],
    box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    box_cls_flattened, box_regression_flattened = [], []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, 1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, 1).reshape(-1, 4)
    return box_cls, box_regression

class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
    """
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': Matcher,
        'fg_bg_sampler': BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }
    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh, pre_nms_top_n, post_nms_top_n,
                 nms_thresh, score_thresh = 0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator, self.head = anchor_generator, head
        self.box_coder = BoxCoder((1.0, 1.0, 1.0, 1.0))
        # used during training
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, True)
        # used during testing
        self._pre_nms_top_n, self._post_nms_top_n = pre_nms_top_n, post_nms_top_n
        self.nms_thresh, self.score_thresh = nms_thresh, score_thresh
        self.l1_loss, self.focal_loss = SmoothL1Loss(reduction = 'none'), FocalLoss(reduction = 'none')

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors: List[Tensor],
        targets: List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        labels, matched_gt_boxes, matched_gt_weights = [], [], []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes, gt_weights = targets_per_image["boxes"], targets_per_image.get('weights')
            device = anchors_per_image.device
            if gt_weights is None:
                gt_weights = torch.ones([gt_boxes.shape[0]], dtype = torch.float32, device = device)
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                matched_gt_boxes_per_image = torch.zeros_like(anchors_per_image)
                labels_per_image = torch.zeros([anchors_per_image.shape[0]],
                                    dtype = anchors_per_image.dtype, device = device)
                weights_per_image = torch.ones([anchors_per_image.shape[0]],
                                    dtype = anchors_per_image.dtype, device = device)
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                clamped_matched_idxs = matched_idxs.clamp(0)
                matched_gt_boxes_per_image = gt_boxes[clamped_matched_idxs]
                weights_per_image = gt_weights[clamped_matched_idxs]
                labels_per_image = (matched_idxs >= 0).to(dtype = anchors_per_image.dtype)
                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                weights_per_image[bg_indices] = 1.0
                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
            matched_gt_weights.append(weights_per_image)
        return labels, matched_gt_boxes, matched_gt_weights

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r, offset = [], 0
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, 1)

    def filter_proposals(self, proposals: Tensor, objectness: Tensor, image_shape: Tuple[int, int],
        num_anchors_per_level: List[int]) -> Tuple[List[Tensor], List[Tensor]]:
        num_images, device = proposals.shape[0], proposals.device
        # do not backprop throught objectness
        objectness = objectness.detach().reshape(num_images, -1)
        levels = [
            torch.full((n,), idx, dtype=torch.int64, device = device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels).reshape(1, -1).expand_as(objectness)
        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = torch.arange(num_images, device = device)
        batch_idx = image_range[:, None]
        objectness = objectness[batch_idx, top_n_idx].sigmoid()
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        final_boxes, final_scores = [], []
        for boxes, scores, lvl in zip(proposals, objectness, levels):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, 1e-3)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor],
        regression_targets: List[Tensor], weights: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        labels, regression_targets = torch.cat(labels), torch.cat(regression_targets)
        weights = torch.cat(weights)
        sampled_pos_inds, sampled_neg_inds = labels >= 1, labels == 0
        sampled_inds = sampled_pos_inds | sampled_neg_inds
        objectness = objectness.flatten()
        box_loss = self.l1_loss(
            pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds])
        pos_num = sampled_pos_inds.sum()
        box_loss = torch.sum(box_loss.sum(-1) * weights[sampled_pos_inds]) / max(1, pos_num)
        objectness_loss = self.focal_loss(objectness[sampled_inds], labels[sampled_inds])
        objectness_loss = torch.sum(objectness_loss * weights[sampled_inds]) / max(1, pos_num)
        return objectness_loss, box_loss

    def forward(self, image_size: Tuple[int, int], features: List[Tensor], targets = None):
        """
        Args:
            image_size (Tuple[int, int]): image size before features are extracted
            features (List[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per image.
            losses (Dict[Tensor]): the losses for the model during training. If, targets is None,
                it is an empty dict.
        """
        # RPN uses all feature maps that are available
        objectness, pred_bbox_deltas = self.head(features)
        anchors = [self.anchor_generator(image_size, [feature.shape[-2:] for feature in features],
                                        features[0].device, features[0].dtype)] * features[0].shape[0]
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, _ = self.filter_proposals(proposals, objectness, image_size, num_anchors_per_level)
        losses = {}
        if targets is not None:
            labels, matched_gt_boxes, weights = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets, weights)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses

class RoIHeads(nn.Module):
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': Matcher
    }
    def __init__(self, box_roi_pool, box_head, box_predictor,
                box_iou_thresh, bbox_reg_weights, score_thresh,
                nms_thresh, detections_per_img):
        super(RoIHeads, self).__init__()
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = [Matcher(iou, iou) for iou in box_iou_thresh]
        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = BoxCoder(bbox_reg_weights)
        self.box_roi_pool, self.box_head, self.box_predictor = box_roi_pool, box_head, box_predictor
        self.score_thresh, self.nms_thresh = score_thresh, nms_thresh
        self.detections_per_img, self.l1_loss = detections_per_img, SmoothL1Loss(reduction = 'none')
        #self.focal_loss = FocalLoss(reduction = 'none', binary = False)
        self.focal_loss = nn.BCEWithLogitsLoss(reduction = 'none')

    def assign_targets_to_proposals(self, proposals: List[Tensor], targets: List[Dict[str, Tensor]],
        proposal_matcher) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        labels, matched_gt_boxes, matched_gt_weights = [], [], []
        for proposal_per_image, targets_per_image in zip(proposals, targets):
            gt_boxes, gt_weights = targets_per_image["boxes"], targets_per_image.get('weights')
            gt_labels = targets_per_image["labels"]
            device = proposal_per_image.device
            if gt_weights is None:
                gt_weights = torch.ones([gt_boxes.shape[0]], dtype = torch.float32, device = device)
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                matched_gt_boxes_per_image = torch.zeros_like(proposal_per_image)
                labels_per_image = torch.zeros([proposal_per_image.shape[0]],
                                    dtype = torch.int64, device = device)
                weights_per_image = torch.ones([proposal_per_image.shape[0]],
                                    dtype = proposal_per_image.dtype, device = device)
            else:
                matched_idxs = proposal_matcher(box_ops.box_iou(gt_boxes, proposal_per_image))
                clamped_matched_idxs = matched_idxs.clamp(0)
                matched_gt_boxes_per_image = gt_boxes[clamped_matched_idxs]
                weights_per_image = gt_weights[clamped_matched_idxs]
                labels_per_image = gt_labels[clamped_matched_idxs].long()
                # Background (negative examples)
                bg_indices = matched_idxs == proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = -1
                weights_per_image[bg_indices] = 1.0
                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -2
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
            matched_gt_weights.append(weights_per_image)
        return labels, matched_gt_boxes, matched_gt_weights

    def postprocess_detections(self, class_logits: Tensor, bbox_regression: Tensor, proposals: List[Tensor],
        image_shape: Tuple[int, int]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        device, num_classes = class_logits.device, class_logits.shape[-1]
        boxes_per_image = [proposal.shape[0] for proposal in proposals]
        pred_boxes_list = self.box_coder.decode(bbox_regression, proposals).split(boxes_per_image)
        pred_scores_list = class_logits.sigmoid().split(boxes_per_image)
        all_boxes, all_scores, all_labels = [], [], []
        score_thresh = torch.as_tensor(self.score_thresh, dtype = torch.float32, device = device)
        for boxes, scores in zip(pred_boxes_list, pred_scores_list):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            # remove predictions with the background label
            # boxes, scores, labels = boxes[:, 1:].reshape(-1, 4), scores[:, 1:], labels[:, 1:].reshape(-1)
            boxes, labels = boxes.reshape(-1, 4), labels.reshape(-1)
            # remove low scoring boxes
            inds = scores > score_thresh
            scores, inds = scores.reshape(-1), inds.reshape(-1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, 1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels

    def fastrcnn_loss(self, class_logits: Tensor, box_regression: Tensor,
        labels: List[Tensor], regression_targets: List[Tensor],
        weights: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Computes the loss for Faster R-CNN.
        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        labels, weights = torch.cat(labels), torch.cat(weights)
        regression_targets = torch.cat(regression_targets)
        sampled_pos_inds_subset, sampled_inds = labels >= 0, labels >= -1
        pos_num = sampled_pos_inds_subset.sum()
        gt_labels = torch.zeros_like(class_logits).scatter_(1, labels.clamp(0)[:, None], 1)
        gt_labels[labels <= -1, :] = 0.0
        cls_loss = self.focal_loss(class_logits[sampled_inds], gt_labels[sampled_inds])
        positive = 1 + torch.relu(gt_labels.sum(0) - 1)
        cls_loss = torch.sum(torch.mean(cls_loss / positive, -1) * weights[sampled_inds])
        labels_pos = labels[sampled_pos_inds_subset]
        box_regression = box_regression.reshape(class_logits.shape[0], box_regression.size(-1) // 4, 4)
        box_loss = self.l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset]
        )
        box_loss = torch.sum(box_loss.sum(-1) * weights[sampled_pos_inds_subset]) / max(1, pos_num)
        return cls_loss, box_loss

    def proposal_correction_loss(self, box_regression: Tensor, labels: List[Tensor],
        regression_targets: List[Tensor], weights: List[Tensor]) -> Tensor:
        """
        Computes the proposal correction loss for Cascade RCNN.
        """
        labels, weights = torch.cat(labels), torch.cat(weights)
        regression_targets = torch.cat(regression_targets)
        sampled_pos_inds_subset = labels > 0
        pos_num = sampled_pos_inds_subset.sum()
        box_loss = self.l1_loss(
            box_regression[sampled_pos_inds_subset], regression_targets[sampled_pos_inds_subset])
        return torch.sum(box_loss.sum(-1) * weights[sampled_pos_inds_subset]) / max(1, pos_num)

    def forward(self, image_size: Tuple[int, int], features: List[Tensor],
                proposals: List[Tensor], targets = None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if targets is not None:
            check_valid_targets(targets)
            # append ground-truth bboxes to proposals
            proposals = [torch.cat((proposal, target['boxes']))
                        for proposal, target in zip(proposals, targets)]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        box_list = []
        for proposal_matcher, box_head, box_predictor in zip(self.proposal_matcher, self.box_head,
                                                        self.box_predictor):
            if targets is not None:
                labels, matched_gt_boxes, weights = self.assign_targets_to_proposals(
                    proposals, targets, proposal_matcher)   
                regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
            box_features = self.box_roi_pool(features, proposals, image_size)
            r = box_predictor(box_head(box_features))
            if isinstance(r, tuple):
                class_logits, box_regression = r
            else:
                class_logits, box_regression = None, r
                proposals = self.box_coder.decode(box_regression, proposals)[:, 0].split(boxes_per_image)
            if targets is not None:
                if class_logits is None:
                    loss_box_reg = self.proposal_correction_loss(box_regression, labels,
                                        regression_targets, weights)
                else:
                    loss_classifier, loss_box_reg = self.fastrcnn_loss(
                        class_logits, box_regression, labels, regression_targets, weights)
                box_list.append(loss_box_reg)
        if targets is not None:
            result = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": sum(box_list) / len(box_list)
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits,
                box_regression, proposals, image_size)
            result = [{"boxes": boxes[i], "labels": labels[i], "scores": scores[i]}
                    for i in range(len(boxes))]
        return result

class FasterRCNNROI(nn.Module):
    def __init__(self, num_levels, output_size, sampling_ratio):
        super(FasterRCNNROI, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.pool = MultiScaleRoIAlign([str(i) for i in range(num_levels)], output_size, sampling_ratio)
        self.output_size = output_size

    def forward(self, features: List[Tensor], proposals: List[Tensor],
        image_size: Tuple[int, int]) -> Tensor:
        return self.pool({str(i): feature for i, feature in enumerate(features)},
                        [p.clone().detach() for p in proposals], [image_size] * len(proposals))

class FasterRCNN(nn.Module):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - weights (Optional[Tensor[N]]): the weights for all boxes

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Args:
        out_channels (int): backbone out_channels.
        num_classes (int): number of output classes of the model (excluding the background).
            If box_predictor is specified, num_classes should be None.
        num_rcnn_stages (int): number of rcnn stages for cascade rcnn.
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (List[nn.Module]): module that takes the cropped feature maps as input
        box_predictor (List[nn.Module]): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (List[float] or float): during inference, only return proposals with a
            classification score greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_iou_thresh (List[float]): IoU between the proposals and the GT box so that they can be
            considered as positive or negative during training of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    """
    def __init__(self, out_channels, num_classes, num_rcnn_stages = 1,
                rpn_anchor_generator = None, rpn_head = None,
                rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
                rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
                rpn_nms_thresh = 0.7, rpn_fg_iou_thresh = 0.7, rpn_bg_iou_thresh = 0.3,
                rpn_score_thresh = 0, box_roi_pool = None, box_head = None, box_predictor = None,
                box_score_thresh = 0.05, box_nms_thresh = 0.5, box_detections_per_img = 100,
                box_iou_thresh = [0.5], bbox_reg_weights = None):
        super(FasterRCNN, self).__init__()
        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head, rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh, rpn_score_thresh)
        if box_roi_pool is None:
            box_roi_pool = FasterRCNNROI(4, 7, 2)
        if box_head is None:
            box_head = [TwoMLPHead(out_channels * box_roi_pool.output_size[0] ** 2,
                        1024) for _ in range(num_rcnn_stages)]
            box_head = nn.Sequential(*box_head)
        if box_predictor is None:
            box_predictor = [CascadeRCNNPredictor(1024) for _ in range(num_rcnn_stages - 1)]
            box_predictor.append(FastRCNNPredictor(1024, num_classes))
            box_predictor = nn.Sequential(*box_predictor)
        assert len(box_head) == len(box_predictor) == len(box_iou_thresh) == num_rcnn_stages
        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_iou_thresh, bbox_reg_weights, box_score_thresh,
            box_nms_thresh, box_detections_per_img)
        self.roi, self.rpn = roi_heads, rpn

    def forward(self, features: List[Tensor], image_size: Tuple[int, int], targets = None):
        """
        Args:
            features (List[Tensor]): feature maps output by the fpn network
            image_size (Tuple[int, int]): image size before feature extraction
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
        proposals, proposal_losses = self.rpn(image_size, features, targets)
        detections = self.roi(image_size, features, proposals, targets)
        losses = proposal_losses
        if targets is None:
            return detections
        else:
            losses.update(detections)
            return losses

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """
    def __init__(self, in_channels, out_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, out_size)
        self.fc7 = nn.Linear(out_size, out_size)
    def forward(self, x):
        return self.fc7(self.fc6(x.flatten(1)).relu()).relu()

class CascadeRCNNPredictor(nn.Module):
    def __init__(self, in_channels):
        super(CascadeRCNNPredictor, self).__init__()
        self.bbox_pred = nn.Linear(in_channels, 4)
    def forward(self, x):
        return self.bbox_pred(x.flatten(1))

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    def forward(self, x):
        x = x.flatten(1)
        return self.cls_score(x), self.bbox_pred(x)