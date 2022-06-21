import torch
from torch.nn import functional as F
from torch import Tensor
from torchvision.ops.boxes import box_area

def maximum(a: Tensor, b: Tensor) -> Tensor:
    return torch.relu(a - b) + b
def minimum(a: Tensor, b: Tensor) -> Tensor:
    return a - torch.relu(a - b)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma = 2, reduction = 'mean',
        with_logits: bool = True, binary: bool = True):
        """
        This module implements multiclass-focal loss.
        Parameters:
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs
                hard examples.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
            with_logits: bool, Whether to pass the input tensor to a softmax function
                before calculating the final loss. Default true for numerical stability.
            binary: True value indicates binary classification, where the predicted values
                are probabilities for the positive class.
        Returns:
            Loss tensor with the reduction option applied.
        """
        assert gamma >= 0 and reduction in ['sum', 'mean', 'none']
        super(FocalLoss, self).__init__()
        self.gamma, self.reduction, self.with_logits = gamma, reduction, with_logits
        self.binary = binary
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Parameters:
            inputs: A float tensor of shape [batch_size, classes, ...] if binary = False
                    or [batch_size, ...] if binary = True. Predictions for each example.
            targets: A long tensor of shape [batch_size, ...].
                    Stores true labels for each element in inputs.
        """
        if self.binary:
            if self.with_logits:
                ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction = 'none')
                prob = inputs.sigmoid()
            else:
                ce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction = 'none')
                prob = inputs
            prob = prob * targets + (1 - prob) * (1 - targets)
            loss = (1 - prob) ** self.gamma * ce_loss
        else:
            if self.with_logits:
                ce_loss = F.cross_entropy(inputs, targets, reduction = 'none', ignore_index = -1)
                prob = inputs.softmax(1)
            else:
                ce_loss = F.nll_loss(inputs, targets, reduction = 'none', ignore_index = -1)
                prob = inputs
            loss = (1 - prob.gather(1, targets[:, None])[:, 0]) ** self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class SmoothL1Loss(torch.nn.Module):
    def __init__(self, beta = 1 / 9, reduction = 'mean'):
        super(SmoothL1Loss, self).__init__()
        self.beta, self.reduction = beta, reduction
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        n = torch.abs(pred - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class GIOULoss(torch.nn.Module):
    def __init__(self, reduction = 'mean'):
        super(GIOULoss, self).__init__()
        self.reduction = reduction
    def foward(self, pred: Tensor, target: Tensor) -> Tensor:
        A_p, A_g = box_area(pred), box_area(target)
        mx, mn = maximum(A_p, A_g), minimum(A_p, A_g)
        I = box_area(torch.cat([mx[:, :2], mn[:, 2:]], 1))
        C = box_area(torch.cat([mn[:, :2], mx[:, 2:]], 1))
        U = A_p + A_g - I
        IOU = I / (U + 1e-6)
        loss = 2 - IOU - U / (C + 1e-6)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss