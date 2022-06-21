import torch
from torch import Tensor

def resize(boxes: Tensor, old_size, new_size) -> Tensor:
    """
    This function resizes boxes from old_size to new_size.
    Parameters:
        boxes: Float Tensor of shape (n, 4), boxes in an image,
            each element of which is (x1, y1, x2, y2).
        old_size: Tuple (H, W), original size of an image.
        new_size: Tuple (H, W), new size of an image.
    Returns:
        Float Tensor of shape (n, 4).
    """
    scale = [new_size[1 - i] / old_size[1 - i] for i in range(2)]
    scale = torch.as_tensor(scale * 2, device = boxes.device)
    return boxes * scale
