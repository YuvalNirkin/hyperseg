import torch
import torch.nn as nn


class MetaSequential(nn.Sequential):
    r"""A sequential container for both regular and meta modules.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """
    def __init__(self, *args):
        super(MetaSequential, self).__init__(*args)
        self.hyper_params = 0
        self._ranges = [0]
        for module in self:
            if hasattr(module, 'hyper_params'):
                self.hyper_params += module.hyper_params
            self._ranges.append(self.hyper_params)

    def forward(self, x, w):
        """ Meta sequential forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            w (torch.Tensor or list of torch.Tensor): Dynamic convolution weights.

        Returns:
            torch.Tensor: Dynamic convolution result.
        """
        w_count = 0
        for i, module in enumerate(self):
            if self._ranges[i] < self._ranges[i + 1]:
                if isinstance(w, (list, tuple)):
                    x = module(x, w[w_count])
                else:
                    x = module(x, w[:, self._ranges[i]:self._ranges[i + 1]].contiguous())
                w_count += 1
            else:
                x = module(x)

        return x
