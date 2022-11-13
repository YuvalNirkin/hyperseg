import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from hyperseg.models.layers.meta_conv import MetaConv2d
from hyperseg.models.layers.meta_sequential import MetaSequential


class MetaPatch(nn.Module):
    """ Make dynamic patch-wise block.

    Args:
        module (nn.Module): Dynamic module to invoke per patch
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    """
    def __init__(self, module: nn.Module, padding=0, padding_mode='reflect'):
        super(MetaPatch, self).__init__()
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")

        self.hyper_module = module
        self.padding = _pair(padding)
        self.padding_mode = padding_mode
        self._padding_repeated_twice = self.padding + self.padding

    @property
    def hyper_params(self):
        return self.hyper_module.hyper_params

    def forward(self, x, weight):
        """ Dynamic patch-wise forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Dynamic weights.

        Returns:
            torch.Tensor: Dynamic patch-wise result.
        """
        b, c, h, w = x.shape
        fh, fw = weight.shape[-2:]
        ph, pw = x.shape[-2] // fh, x.shape[-1] // fw
        kh, kw = ph + self.padding[0] * 2, pw + self.padding[1] * 2
        weight = weight.permute(0, 2, 3, 1).reshape(-1, weight.shape[1]).contiguous()
        x = F.pad(x, self._padding_repeated_twice, mode=self.padding_mode)
        x = torch.nn.functional.unfold(x, (kh, kw), stride=(ph, pw))  # B x (C x (ph x pw)) x (fh * fw)
        x = x.transpose(1, 2).reshape(-1, c, kh, kw).contiguous()
        x = self.hyper_module(x, weight)
        x = x.view(b, fh * fw, -1, ph * pw).permute(0, 2, 3, 1).reshape(b, -1, fh * fw)
        x = F.fold(x, (h, w), kernel_size=(ph, pw), stride=(ph, pw))

        return x


class MetaPatchConv2d(MetaPatch):
    r"""Applies a dynamic patch-wise 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    Note:

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = MetaPatchConv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = MetaPatchConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = MetaPatchConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='reflect'):
        conv = MetaConv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups)
        super(MetaPatchConv2d, self).__init__(conv, padding, padding_mode)

    @property
    def in_channels(self):
        return self.hyper_module.in_channels

    @property
    def out_channels(self):
        return self.hyper_module.out_channels

    @property
    def kernel_size(self):
        return self.hyper_module.kernel_size

    @property
    def groups(self):
        return self.hyper_module.groups

    def __repr__(self):
        s = self.__class__.__name__ + '({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.hyper_module.dilation != (1,) * len(self.hyper_module.dilation):
            s += ', dilation={dilation}'
        if self.hyper_module.groups != 1:
            s += ', groups={groups}'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ')'
        d = {**self.hyper_module.__dict__}
        d['padding'] = self.padding
        d['padding_mode'] = self.padding_mode
        return s.format(**d)


def make_meta_patch_conv2d_block(in_nc, out_nc, kernel_size=3, stride=1, padding=None, dilation=1, groups=1,
                                 padding_mode='reflect', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True),
                                 dropout=None):
    """ Defines a meta patch-wise convolution block with a normalization layer, an activation layer, and an optional
    dropout layer.

    Args:
        in_nc (int): Input number of channels
        out_nc (int): Output number of channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int, optional): The amount of padding for the height and width dimensions
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        dropout (float): If specified, enables dropout with the given probability
    """
    assert dropout is None or isinstance(dropout, float)
    padding = kernel_size // 2 if padding is None else padding
    layers = [MetaPatchConv2d(in_nc, out_nc, kernel_size, stride, padding, dilation, groups, padding_mode)]
    if norm_layer is not None:
        layers.append(norm_layer(out_nc))
    if act_layer is not None:
        layers.append(act_layer)
    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return MetaSequential(*layers)


def main(model='hyperseg.models.layers.meta_conv.MetaConv2d(kernel_size=3)', in_channels=10, out_channels=20,
         padding=0, test_fps=False):
    from hyperseg.utils.obj_factory import obj_factory
    from hyperseg.utils.utils import set_device
    import time
    from tqdm import tqdm

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    device, gpus = set_device()
    model = obj_factory(model, in_channels=in_channels, out_channels=out_channels).to(device)
    patch_model = MetaPatch(model, padding=padding)

    x = torch.rand(2, in_channels, 256, 256).to(device)
    w = torch.ones(2, model.hyper_params, 8, 8).to(device)
    out = patch_model(x, w)
    print(out.shape)

    if test_fps:
        total_time = 0.
        total_iterations = 0
        pbar = tqdm(range(1000), unit='frames')
        for i in pbar:
            # Start measuring time
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            out = patch_model(x[:1], w[:1])

            # Stop measuring time
            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            total_time += elapsed_time
            total_iterations += out.shape[0]
            fps = total_iterations / total_time

            # Update progress bar info
            pbar.set_description(f'fps = {fps}')


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='hyperseg.models.layers.meta_conv.MetaConv2d(kernel_size=3)',
                        help='model object')
    parser.add_argument('-ic', '--in_channels', default=10, type=int,
                        metavar='N', help='number of input features')
    parser.add_argument('-oc', '--out_channels', default=20, type=int,
                        metavar='N', help='number of output features')
    parser.add_argument('-p', '--padding', default=0, type=int,
                        metavar='N', help='amount of padding for each spatial dimension')
    parser.add_argument('-tf', '--test_fps', action='store_true',
                        help='test fps')
    main(**vars(parser.parse_args()))
