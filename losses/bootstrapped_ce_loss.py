import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Adapted from: https://github.com/PingoLH/FCHarDNet/blob/master/ptsemseg/loss/loss.py
class BootstrappedCrossEntropyLoss(nn.CrossEntropyLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    """
    def __init__(self, k=4096, thresh=0.3, weight: Optional[torch.Tensor] = None, ignore_index: int = -100,
                 reduction: str = 'mean') -> None:
        super(BootstrappedCrossEntropyLoss, self).__init__(weight, None, ignore_index, None, 'none')
        self.k = k
        self.thresh = thresh
        self.topk_reduction = reduction

    def _bootstrap_xentropy_single(self, input, target):
        input = input.permute(0, 2, 3, 1).reshape(-1, input.shape[1])
        target = target.view(-1)
        loss = super(BootstrappedCrossEntropyLoss, self).forward(input, target)
        sorted_loss, _ = torch.sort(loss, descending=True)
        loss = sorted_loss[sorted_loss > self.thresh] if sorted_loss[self.k] > self.thresh else sorted_loss[:self.k]
        reduced_topk_loss = torch.mean(loss)

        return reduced_topk_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0

        # Bootstrap from each image not entire batch
        for i in range(input.shape[0]):
            loss += self._bootstrap_xentropy_single(input[i].unsqueeze(0), target[i].unsqueeze(0))

        return loss / float(input.shape[0])
