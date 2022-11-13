from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    """ Learning rate policy where current learning rate of each
    parameter group equals to the initial learning rate multiplying
    by :math:`(1 - \frac{last_epoch}{max_epoch})^power`.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_epoch (int): Period of learning rate decay.
        power (float): Power factor of learning rate multiplier decay. Default: 0.9.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * ((1.0 - float(self.last_epoch) / float(self.max_epoch)) ** self.power)
                for base_lr in self.base_lrs]
