from ..models.detection.utils import AnchorGenerator, Matcher
from typing import Tuple, List
import torch
from torch import Tensor
from torchvision.ops import box_iou

class anchor_matcher():
    def __init__(self, anchor_sizes, aspect_ratios, fg_iou, bg_iou, fixed_size):
        """
        This module computes the anchor matching rate given parameters.
        Parameters:
            anchor_sizes (List[Tuple[int, int...]]): anchor sizes fed into the anchor generator
                like that in faster rcnn. Each element needs to be a power of 2.
            aspect_ratios: aspect ratios fed into the anchor generator like that in faster rcnn.
            fg_iou: iou threshold for an anchor to be considered a foreground anchor.
            bg_iou: iou threshold for an anchor to be considered a background anchor.
            fixed_size (bool): Whether the image size and feature sizes will be fixed.
        """
        self.generator = AnchorGenerator(anchor_sizes, aspect_ratios, fixed_size)
        self.proposal_matcher = Matcher(fg_iou, bg_iou)

    def match(self, gt_boxes: Tensor, image_size: Tuple[int, int], feature_sizes: List[Tuple[int, int]]):
        """
        This function returns a matched_idxs indicating which boxes are matched.
        Parameters:
            gt_boxes (Tensor): a tensor of shape (n, 4) representing the ground-truth boxes in an image.
            image_size (Tuple[int, int]): size of the input image.
            feature_sizes (List[Tuple[int, int]]): sizes of feature maps output by the fpn network.
        """
        anchors = self.generator(image_size, feature_sizes, device = gt_boxes.device)
        match_quality_matrix = box_iou(gt_boxes, anchors)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        matched_boxes = torch.zeros([gt_boxes.shape[0]], dtype = torch.int64, device = gt_boxes.device)
        matched_boxes[matched_idxs[matched_idxs >= 0]] = 1
        return torch.where(matched_boxes)[0]