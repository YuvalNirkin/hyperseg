import numbers
import numpy as np
from itertools import groupby
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from hyperseg.models.layers.meta_conv import MetaConv2d
from hyperseg.models.layers.meta_sequential import MetaSequential


class HyperGen(nn.Module):
    """ Hypernetwork generator comprised of a backbone network, weight mapper, and a decoder.

    Args:
        backbone (nn.Module factory): Backbone network
        weight_mapper (nn.Module factory): Weight mapper network.
        in_nc (int): input number of channels.
        num_classes (int): output number of classes.
        kernel_sizes (int): the kernel size of the decoder layers.
        level_layers (int): number of layers in each level of the decoder.
        level_channels (list of int, optional): If specified, sets the output channels of each level in the decoder.
        expand_ratio (int): inverted residual block's expansion ratio in the decoder.
        groups (int, optional): Number of blocked connections from input channels to output channels.
        weight_groups (int, optional): per level signal to weights groups in the decoder.
        inference_hflip (bool): If true, enables horizontal flip of input tensor.
        inference_gather (str): Inference gather type: ``mean'' or ``max''.
        with_out_fc (bool): If True, add a final fully connected layer to the decoder.
        decoder_groups (int, optional): per level groups in the decoder.
        decoder_dropout (float): If specified, enables dropout with the given probability.
        coords_res (list of tuple of int, optional): list of inference resolutions for caching positional embedding.
    """
    def __init__(self, backbone, weight_mapper, in_nc=3, num_classes=3, kernel_sizes=3, level_layers=1,
                 level_channels=None, expand_ratio=1, groups=1, weight_groups=1, inference_hflip=False,
                 inference_gather='mean', with_out_fc=False, decoder_groups=1, decoder_dropout=None, coords_res=None):
        super(HyperGen, self).__init__()
        self.inference_hflip = inference_hflip
        self.inference_gather = inference_gather

        self.backbone = backbone()
        feat_channels = [in_nc] + self.backbone.feat_channels[:-1]
        self.decoder = MultiScaleDecoder(feat_channels, self.backbone.feat_channels[-1], num_classes, kernel_sizes,
                                         level_layers, level_channels, with_out_fc=with_out_fc, out_kernel_size=1,
                                         expand_ratio=expand_ratio, groups=decoder_groups, weight_groups=weight_groups,
                                         dropout=decoder_dropout, coords_res=coords_res)
        self.weight_mapper = weight_mapper(self.backbone.feat_channels[-1], self.decoder.param_groups)

    @property
    def hyper_params(self):
        return self.decoder.hyper_params

    def process_single_tensor(self, x, hflip=False):
        x = torch.flip(x, [-1]) if hflip else x
        features = self.backbone(x)
        weights = self.weight_mapper(features[-1])
        x = [x] + features[:-1]
        x = self.decoder(x, weights)
        x = torch.flip(x, [-1]) if hflip else x

        return x

    def gather_results(self, x, y=None):
        assert x is not None
        if y is None:
            return x
        if self.inference_gather == 'mean':
            return (x + y) * 0.5
        else:
            return torch.max(x, y)

    def forward(self, x):
        assert isinstance(x, (list, tuple, torch.Tensor)), f'x must be of type list, tuple, or tensor'
        if isinstance(x, torch.Tensor):
            return self.process_single_tensor(x)

        # Note: the first pyramid will determine the output resolution
        out_res = x[0].shape[2:]
        out = None
        for p in x:
            if self.inference_hflip:
                p = torch.max(self.process_single_tensor(p), self.process_single_tensor(p, hflip=True))
            else:
                p = self.process_single_tensor(p)

            # Resize current image to output resolution if necessary
            if p.shape[2:] != out_res:
                p = F.interpolate(p, out_res, mode='bilinear', align_corners=False)

            out = self.gather_results(p, out)

        return out


class MultiScaleDecoder(nn.Module):
    """ Dynamic multi-scale decoder.

    Args:
        feat_channels (list of int): per level input feature channels.
        signal_channels (list of int): per level input signal channels.
        num_classes (int): output number of classes.
        kernel_sizes (int): the kernel size of the layers.
        level_layers (int): number of layers in each level.
        level_channels (list of int, optional): If specified, sets the output channels of each level.
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        out_kernel_size (int): kernel size of the final output layer.
        expand_ratio (int): inverted residual block's expansion ratio.
        groups (int, optional): number of blocked connections from input channels to output channels.
        weight_groups (int, optional): per level signal to weights.
        with_out_fc (bool): If True, add a final fully connected layer.
        dropout (float): If specified, enables dropout with the given probability.
        coords_res (list of tuple of int, optional): list of inference resolutions for caching positional embedding.
    """
    def __init__(self, feat_channels, signal_channels, num_classes=3, kernel_sizes=3, level_layers=1,
                 level_channels=None, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6(inplace=True), out_kernel_size=1,
                 expand_ratio=1, groups=1, weight_groups=1, with_out_fc=False, dropout=None, coords_res=None):
        super(MultiScaleDecoder, self).__init__()
        if isinstance(kernel_sizes, numbers.Number):
            kernel_sizes = (kernel_sizes,) * len(level_channels)
        if isinstance(level_layers, numbers.Number):
            level_layers = (level_layers,) * len(level_channels)
        if isinstance(expand_ratio, numbers.Number):
            expand_ratio = (expand_ratio,) * len(level_channels)
        assert len(kernel_sizes) == len(level_channels), \
            f'kernel_sizes ({len(kernel_sizes)}) must be of size {len(level_channels)}'
        assert len(level_layers) == len(level_channels), \
            f'level_layers ({len(level_layers)}) must be of size {len(level_channels)}'
        assert len(expand_ratio) == len(level_channels), \
            f'expand_ratio ({len(expand_ratio)}) must be of size {len(level_channels)}'
        if isinstance(groups, (list, tuple)):
            assert len(groups) == len(level_channels), f'groups ({len(groups)}) must be of size {len(level_channels)}'
        self.level_layers = level_layers
        self.levels = len(level_channels)
        self.layer_params = []
        feat_channels = feat_channels[::-1]  # Reverse the order of the feature channels
        self.coords_cache = {}
        self.weight_groups = weight_groups

        # For each level
        prev_channels = 0
        for level in range(self.levels):
            curr_ngf = feat_channels[level]
            curr_out_ngf = curr_ngf if level_channels is None else level_channels[level]
            prev_channels += curr_ngf  # Accommodate the previous number of channels
            curr_layers = []
            kernel_size = kernel_sizes[level]

            # For each layer in the current level
            for layer in range(self.level_layers[level]):
                if (not with_out_fc) and (level == (self.levels - 1) and (layer == (self.level_layers[level] - 1))):
                    curr_out_ngf = num_classes
                if kernel_size > 1:
                    curr_layers.append(HyperPatchInvertedResidual(
                        prev_channels + 2, curr_out_ngf, kernel_size, expand_ratio=expand_ratio[level],
                        norm_layer=norm_layer, act_layer=act_layer))
                else:
                    group = groups[level] if isinstance(groups, (list, tuple)) else groups
                    curr_layers.append(make_hyper_patch_conv2d_block(prev_channels + 2, curr_out_ngf, kernel_size,
                                                                     groups=group))
                prev_channels = curr_out_ngf

            # Add level layers to module
            self.add_module(f'level_{level}', MetaSequential(*curr_layers))

        # Add the last layer
        if with_out_fc:
            out_fc_layers = [nn.Dropout2d(dropout, True)] if dropout is not None else []
            out_fc_layers.append(
                HyperPatchConv2d(prev_channels, num_classes, out_kernel_size, padding=out_kernel_size // 2))
            self.out_fc = MetaSequential(*out_fc_layers)
        else:
            self.out_fc = None

        # Calculate number of hyper parameters, weight ranges, and total number of hyper parameters per level
        self.hyper_params = 0
        self._ranges = [0]
        self.param_groups = []
        for level in range(self.levels):
            level_layers = getattr(self, f'level_{level}')
            self.hyper_params += level_layers.hyper_params
            self._ranges.append(self.hyper_params)
            self.param_groups.append(level_layers.hyper_params)
        if with_out_fc:
            self.hyper_params += self.out_fc.hyper_params
            self.param_groups.append(self.out_fc.hyper_params)
        self._ranges.append(self.hyper_params)

        # Cache image coordinates
        if coords_res is not None:
            for res in coords_res:
                res_pyd = [(res[0] // 2 ** i, res[1] // 2 ** i) for i in range(self.levels)]
                for level_res in res_pyd:
                    self.register_buffer(f'coord{level_res[0]}_{level_res[1]}',
                                         self.cache_image_coordinates(*level_res))

        # Initialize signal to weights
        hyper_params = get_hyper_params(self)
        min_unit = max(weight_groups)
        signal_features = divide_feature(signal_channels, hyper_params, min_unit=min_unit)
        init_signal2weights(self, list(signal_features), weight_groups=weight_groups)
        self.hyper_params = sum(hyper_params)

    def cache_image_coordinates(self, h, w):
        x = torch.linspace(-1, 1, steps=w)
        y = torch.linspace(-1, 1, steps=h)
        grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0).unsqueeze(0)

        return grid

    def get_image_coordinates(self, b, h, w, device):
        cache = f'coord{h}_{w}'
        if hasattr(self, cache):
            return getattr(self, cache).expand(b, -1, -1, -1)

        x = torch.linspace(-1, 1, steps=w, device=device)
        y = torch.linspace(-1, 1, steps=h, device=device)
        grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0).unsqueeze(0)

        return grid.expand(b, -1, -1, -1)

    def forward(self, x, s):
        # assert isinstance(w, (list, tuple))
        # assert len(x) <= self.levels

        # For each level
        p = None
        for level in range(self.levels):
            level_layers = getattr(self, f'level_{level}')

            # Initial layer input
            if p is None:
                p = x[-level - 1]
            else:
                # p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample x2
                if p.shape[2:] != x[-level - 1].shape[2:]:
                    p = F.interpolate(p, x[-level - 1].shape[2:], mode='bilinear', align_corners=False)  # Upsample
                p = torch.cat((x[-level - 1], p), dim=1)

            # Add image coordinates
            p = torch.cat([self.get_image_coordinates(p.shape[0], *p.shape[-2:], p.device), p], dim=1)

            # Computer the output for the current level
            p = level_layers(p, s)

        # Last layer
        if self.out_fc is not None:
            p = self.out_fc(p, s)

        # Upscale the prediction the finest feature map resolution
        if p.shape[2:] != x[0].shape[2:]:
            p = F.interpolate(p, x[0].shape[2:], mode='bilinear', align_corners=False)  # Upsample

        return p


def get_hyper_params(model):
    hyper_params = []

    # For each child module
    for name, m in model.named_children():
        if isinstance(m, (HyperPatchConv2d, HyperPatchNoPadding, HyperPatchInvertedResidual)):
            hyper_params.append(m.hyper_params)
        else:
            hyper_params += get_hyper_params(m)

    return hyper_params


def init_signal2weights(model, signal_features, signal_index=0, weight_groups=1):
    # For each child module
    for name, m in model.named_children():
        if isinstance(m, (HyperPatchConv2d, HyperPatchNoPadding, HyperPatchInvertedResidual)):
            curr_feature_nc = signal_features.pop(0)
            curr_weight_group = weight_groups.pop(0) if isinstance(weight_groups, list) else weight_groups
            m.init_signal2weights(curr_feature_nc, signal_index, curr_weight_group)
            signal_index += curr_feature_nc
        else:
            init_signal2weights(m, signal_features, signal_index, weight_groups)


class HyperPatchInvertedResidual(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, expand_ratio=1, norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6(inplace=True), padding_mode='reflect'):
        super(HyperPatchInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.padding_mode = padding_mode
        self.padding = (1, 1)
        self._padding_repeated_twice = self.padding + self.padding
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.kernel_size = _pair(kernel_size)
        self.hidden_dim = int(round(in_nc * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_nc == out_nc
        self.act_layer = act_layer
        self.bn1 = norm_layer(self.hidden_dim)
        self.bn2 = norm_layer(self.hidden_dim)
        self.bn3 = norm_layer(self.out_nc)

        # Calculate hyper params and weight ranges
        self.hyper_params = 0
        self._ranges = [0]
        self.hyper_params += in_nc * self.hidden_dim
        self._ranges.append(self.hyper_params)
        self.hyper_params += np.prod((self.hidden_dim,) + self.kernel_size)
        self._ranges.append(self.hyper_params)
        self.hyper_params += self.hidden_dim * out_nc
        self._ranges.append(self.hyper_params)

        self.signal_channels = None
        self.signal_index = None
        self.signal2weights = None

    def init_signal2weights(self, signal_channels, signal_index=0, groups=1):
        self.signal_channels = signal_channels
        self.signal_index = signal_index
        weight_channels = next_multiply(self.hyper_params, groups)
        self.signal2weights = nn.Conv2d(signal_channels, weight_channels, 1, bias=False, groups=groups)

    def apply_signal2weights(self, s):
        if self.signal2weights is None:
            return s
        w = self.signal2weights(s[:, self.signal_index:self.signal_index + self.signal_channels])[:, :self.hyper_params]

        return w

    def conv(self, x, s):
        weight = self.apply_signal2weights(s)
        b, c, h, w = x.shape
        # assert b == 1
        fh, fw = weight.shape[-2:]
        ph, pw = x.shape[-2] // fh, x.shape[-1] // fw
        kh, kw = ph + self.padding[0] * 2, pw + self.padding[1] * 2

        if self.padding_mode != 'zeros' and np.any(self._padding_repeated_twice):
            x = F.pad(x, self._padding_repeated_twice, mode=self.padding_mode)
            padding = _pair(0)
        else:
            padding = self.padding

        x = x.permute(0, 2, 3, 1).unfold(1, kh, ph).unfold(2, kw, pw).reshape(1, -1, kh, kw)

        if b == 1:
            weight = weight.permute(0, 2, 3, 1).view(-1, weight.shape[1])
        else:
            weight = weight.permute(0, 2, 3, 1).reshape(-1, weight.shape[1])

        # Conv1
        weight1 = weight[:, self._ranges[0]:self._ranges[1]].reshape(b * fh * fw * self.hidden_dim, self.in_nc, 1, 1)
        x = F.conv2d(x, weight1, bias=None, groups=b * fh * fw)
        x = self.bn1(x.view(b * fh * fw, -1, kh, kw)).view(1, -1, kh, kw)
        x = self.act_layer(x)
        # x = self.act_layer(self.bn1(F.conv2d(x, weight1, bias=None, groups=b * fh * fw)))

        # Conv2
        weight2 = weight[:, self._ranges[1]:self._ranges[2]].reshape(b * fh * fw * self.hidden_dim, 1,
                                                                     *self.kernel_size)
        x = F.conv2d(x, weight2, bias=None, stride=self.stride, groups=b * fh * fw * self.hidden_dim)
        x = self.bn2(x.view(b * fh * fw, -1, ph, pw)).view(1, -1, ph, pw)
        x = self.act_layer(x)

        # Conv3
        weight3 = weight[:, self._ranges[2]:self._ranges[3]].reshape(b * fh * fw * self.out_nc, self.hidden_dim, 1, 1)
        x = F.conv2d(x, weight3, bias=None, groups=b * fh * fw)
        x = self.bn3(x.view(b * fh * fw, -1, ph, pw))

        x = x.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)

        return x

    def forward(self, x, s):
        if self.use_res_connect:
            return x + self.conv(x, s)
        else:
            return self.conv(x, s)


class WeightMapper(nn.Module):
    """ Weight mapper module (called context head in the paper).

    Args:
        in_channels (int): input number of channels.
        out_channels (int): output number of channels.
        levels (int): number of levels operating on different strides.
        bias (bool): if True, enables bias in all convolution operations.
        min_unit (int): legacy parameter, no longer used.
        weight_groups (int): legacy parameter, no longer used.
    """
    def __init__(self, in_channels, out_channels, levels=3, bias=False, min_unit=4, weight_groups=1):
        super(WeightMapper, self).__init__()
        assert levels > 0, 'levels must be greater than zero'
        assert in_channels % 2 == 0, 'in_channels must be divisible by 2'
        if isinstance(weight_groups, (list, tuple)):
            assert len(weight_groups) == len(out_channels), \
                f'groups ({len(weight_groups)}) must be of size {len(out_channels)}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.bias = bias
        self.weight_groups = weight_groups

        # Add blocks
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True))

        for level in range(self.levels - 1):
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, bias=bias),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True)))
            self.up_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 1, bias=bias),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True)))

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.in_conv(x)

        # Down stream
        feat = [x]
        for level in range(self.levels - 1):
            feat.append(self.down_blocks[level](feat[-1]))

        # Average the last feature map
        orig_shape = feat[-1].shape
        if orig_shape[-2:] != (1, 1):
            x = F.adaptive_avg_pool2d(feat[-1], 1)
            x = F.interpolate(x, orig_shape[-2:], mode='nearest')

        # Up stream
        for level in range(self.levels - 2, -1, -1):
            x = torch.cat((feat.pop(-1), x), dim=1)
            x = self.up_blocks[level](x)
            x = self.upsample(x)

        # Output head
        x = torch.cat((feat.pop(-1), x), dim=1)

        return x


def next_multiply(x, base):
    return type(x)(np.ceil(x / base) * base)


class HyperPatchNoPadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super(HyperPatchNoPadding, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.hyper_params = np.prod((out_channels, in_channels // groups) + self.kernel_size)
        self.signal_channels = None
        self.signal_index = None
        self.signal2weights = None

    def init_signal2weights(self, signal_channels, signal_index=0, groups=1):
        self.signal_channels = signal_channels
        self.signal_index = signal_index
        weight_channels = next_multiply(self.hyper_params, groups)
        self.signal2weights = nn.Conv2d(signal_channels, weight_channels, 1, bias=False, groups=groups)

    def apply_signal2weights(self, s):
        if self.signal2weights is None:
            return s
        w = self.signal2weights(s[:, self.signal_index:self.signal_index + self.signal_channels])[:, :self.hyper_params]

        return w

    def forward(self, x, s):
        weight = self.apply_signal2weights(s)
        b, c, h, w = x.shape
        fh, fw = weight.shape[-2:]
        ph, pw = x.shape[-2] // fh, x.shape[-1] // fw

        weight = weight.permute(0, 2, 3, 1).reshape(
            b * fh * fw * self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        x = x.view(b, c, fh, ph, fw, pw).permute(0, 2, 4, 1, 3, 5).reshape(1, -1, ph, pw)
        x = F.conv2d(x, weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b * fh * fw * self.groups)
        x = x.view(b, fh, fw, -1, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, -1, h, w)

        return x


class HyperPatch(nn.Module):
    """ Make dynamic patch-wise block.

    Args:
        module (nn.Module): Dynamic module to invoke per patch
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    """
    def __init__(self, module: nn.Module, padding=0, padding_mode='reflect'):
        super(HyperPatch, self).__init__()
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")

        self.hyper_module = module
        self.padding = _pair(padding)
        self.padding_mode = padding_mode
        self._padding_repeated_twice = self.padding + self.padding

        self.signal_channels = None
        self.signal_index = None
        self.signal2weights = None

    @property
    def hyper_params(self):
        return self.hyper_module.hyper_params

    def init_signal2weights(self, signal_channels, signal_index=0, groups=1):
        self.signal_channels = signal_channels
        self.signal_index = signal_index
        self.signal2weights = nn.Conv2d(signal_channels, self.hyper_params, 1, bias=False, groups=groups)

    def apply_signal2weights(self, s):
        if self.signal2weights is None:
            return s
        w = self.signal2weights(s[:, self.signal_index:self.signal_index + self.signal_channels])[:, :self.hyper_params]

        return w

    def forward(self, x, s):
        weight = self.apply_signal2weights(s)
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


class HyperPatchConv2d(HyperPatch):
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
        >>> m = HyperPatchConv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = HyperPatchConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = HyperPatchConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
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
        super(HyperPatchConv2d, self).__init__(conv, padding, padding_mode)

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


def make_hyper_patch_conv2d_block(in_nc, out_nc, kernel_size=3, stride=1, padding=None, dilation=1, groups=1,
                                  padding_mode='reflect', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True),
                                  dropout=None):
    """ Defines a hyper patch-wise convolution block with a normalization layer, an activation layer, and an optional
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
    if padding == 0:
        layers = [HyperPatchNoPadding(in_nc, out_nc, kernel_size, stride, dilation, groups)]
    else:
        layers = [HyperPatchConv2d(in_nc, out_nc, kernel_size, stride, padding, dilation, groups, padding_mode)]
    if norm_layer is not None:
        layers.append(norm_layer(out_nc))
    if act_layer is not None:
        layers.append(act_layer)
    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return MetaSequential(*layers)


def divide_feature(in_feature, out_features, min_unit=8):
    """ Divides in_feature relative to each of the provided out_features.

    The division of the input feature will be in multiplies of "min_unit".
    The algorithm makes sure that equal output features will get the same portion of the input feature.
    The smallest out feature will receive all the round down overflow (usually the final fc)

    Args:
        in_feature: the input feature to divide
        out_features: the relative sizes of the output features
        min_unit: each division of the input feature will be divisible by this number.
        in_feature must be divisible by this number as well

    Returns:
        np.array: array of integers of the divided input feature in the size of out_features.
    """
    assert in_feature % min_unit == 0, f'in_feature ({in_feature}) must be divisible by min_unit ({min_unit})'
    units = in_feature // min_unit
    indices = np.argsort(out_features)
    out_features_sorted = np.array(out_features)[indices]
    out_feat_groups = [(k, indices[list(g)]) for k, g in groupby(range(len(indices)), lambda i: out_features_sorted[i])]
    out_feat_groups.sort(key=lambda x: x[0] * len(x[1]), reverse=True)
    units_feat_ratio = float(units) / sum(out_features)

    # For each feature group
    out_group_units = [len(out_feat_group[1]) for out_feat_group in out_feat_groups]
    remaining_units = units - sum(out_group_units)
    for i, out_feat_group in enumerate(out_feat_groups):    # out_feat_group: (out_feature, indices array)
        if i < (len(out_feat_groups) - 1):
            n = len(out_feat_group[1])  # group size
            curr_out_feat_size = out_feat_group[0] * n
            curr_units = max(curr_out_feat_size * units_feat_ratio, n)
            curr_units = curr_units // n * n - n  # Make divisible by num elements
            curr_units = min(curr_units, remaining_units)
            out_group_units[i] += curr_units
            remaining_units -= curr_units
            if remaining_units == 0:
                break
        else:
            out_group_units[-1] += remaining_units

    # Final feature division
    divided_in_features = np.zeros(len(out_features), dtype=int)
    for i, out_feat_group in enumerate(out_feat_groups):
        for j in range(len(out_feat_group[1])):
            divided_in_features[out_feat_group[1][j]] = out_group_units[i] // len(out_feat_group[1]) * min_unit

    return divided_in_features


def hyperseg_efficientnet(model_name, pretrained=False, out_feat_scale=0.25, levels=3, weights_path=None, **kwargs):
    from hyperseg.models.backbones.efficientnet import efficientnet
    from functools import partial

    weight_mapper = partial(WeightMapper, levels=levels)
    backbone = partial(efficientnet, model_name, pretrained=pretrained, out_feat_scale=out_feat_scale, head=None,
                       return_features=True)
    model = HyperGen(backbone, weight_mapper, **kwargs)

    if weights_path is not None:
        checkpoint = torch.load(weights_path)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=True)

    return model


def main(model='hyperseg.models.hyperseg_v1_0.hypergen_efficientnet', res=(512,),
         pyramids=None,
         train=False):
    from hyperseg.utils.obj_factory import obj_factory
    from hyperseg.utils.utils import set_device
    from hyperseg.utils.img_utils import create_pyramid
    from tqdm import tqdm

    assert len(res) <= 2, f'res must be either a single number or a pair of numbers: "{res}"'
    res = res * 2 if len(res) == 1 else res

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    device, gpus = set_device()
    model = obj_factory(model).to(device).train(train)
    x = torch.rand(1, 3, *res).to(device)
    x = create_pyramid(x, pyramids) if pyramids is not None else x
    pred = model(x)
    print(pred.shape)


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('hyperseg test')
    parser.add_argument('-m', '--model',
                        default='hyperseg.models.hyperseg_v1_0.hypergen_efficientnet',
                        help='model object')
    parser.add_argument('-r', '--res', default=(512,), type=int, nargs='+',
                        metavar='N', help='image resolution')
    parser.add_argument('-p', '--pyramids', type=int, metavar='N',
                        help='number of image pyramids')
    parser.add_argument('-t', '--train', action='store_true',
                        help='If True, sets the model to training mode')
    main(**vars(parser.parse_args()))
