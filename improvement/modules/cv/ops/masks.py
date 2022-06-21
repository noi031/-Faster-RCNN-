from torch import Tensor

def mask_iou(masks1: Tensor, masks2: Tensor) -> Tensor:
    n1, n2 = masks1.shape[0], masks2.shape[0]
    masks1 = masks1.contiguous().view(n1, -1).float()
    masks2 = masks2.contiguous().view(n2, -1).float()
    inter = masks1 @ masks2.T
    masks1, masks2 = masks1.sum(1)[:, None], masks2.sum(1)[None]
    union = masks1 + masks2 - inter
    return inter / (union + 1e-6)