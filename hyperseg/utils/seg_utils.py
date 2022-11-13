import numpy as np
import torch


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        with torch.no_grad():
            n = self.num_classes
            if self.mat is None:
                self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        with torch.no_grad():
            h = self.mat.float()
            acc_global = torch.diag(h).sum() / h.sum()

            # TODO: Make sure this is correct
            h_sum1 = h.sum(1)
            # h_sum1[h_sum1 == 0.] = 1.   # Avoid division by 0
            acc = torch.diag(h) / (h_sum1 + 1e-6)
            iu = torch.diag(h) / (h_sum1 + h.sum(0) - torch.diag(h) + 1e-6)

            # acc = torch.diag(h) / h.sum(1)
            # iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


class IOUBenchmark(object):
    def __init__(self, num_classes=None):
        self.confmat = None if num_classes is None else ConfusionMatrix(num_classes)

    def reset(self):
        # self.confmat = None
        if self.confmat is not None:
            self.confmat.reset()

    def to(self, device):
        return self

    def __call__(self, pred, target):
        if self.confmat is None:
            assert pred.dim() == 4, 'prediction must be of 4 dimensions if num_classes was not specified'
            self.confmat = ConfusionMatrix(pred.shape[1])
        self.confmat.update(target.flatten(), pred.argmax(1).flatten() if pred.dim() == 4 else pred.flatten())
        acc_global, acc, iou = self.confmat.compute()
        miou = iou.mean().item()

        return {'iou': miou}


def blend_seg(img, seg, color_map=None, alpha=0.5, ignore_index=0):
    """ Blend images with their corresponding segmentation prediction.

    Args:
        img (torch.Tensor): A batch of image tensors of shape (B, 3, H, W) where B is the batch size,
            H is the images height and W is the images width
        seg (torch.Tensor): A batch of segmentation predictions of shape (B, C, H, W) where B is the batch size,
            C is the number of segmentation classes, H is the images height and W is the images width
        alpha: alpha (float): Opacity value for the segmentation in the range [0, 1] where 0 is completely transparent
            and 1 is completely opaque

    Returns:
        torch.Tensor: The blended image.
    """
    # color_map_tensor = torch.from_numpy(color_map.astype('float32')).to(img.device).div_(128.).sub_(1.)
    color_map_tensor = torch.from_numpy(np.array(color_map, dtype='float32')).to(img.device).div_(128.).sub_(1.)
    seg_classes = seg.argmax(1) if seg.dim() == 4 else seg.clone()
    seg_classes[seg_classes >= color_map_tensor.shape[0]] = ignore_index
    seg_rgb = color_map_tensor[seg_classes].permute(0, 3, 1, 2)
    alpha_mask = (1. - (seg_classes != ignore_index).float() * alpha).unsqueeze(1).repeat(1, 3, 1, 1)

    return img * alpha_mask + seg_rgb * (1. - alpha_mask)
