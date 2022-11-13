from itertools import groupby
import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperseg.models.layers.meta_sequential import MetaSequential
from hyperseg.models.layers.meta_patch import MetaPatchConv2d, make_meta_patch_conv2d_block


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
        weight_groups (int, optional): per level signal to weights in the decoder.
        inference_hflip (bool): If true, enables horizontal flip of input tensor.
        inference_gather (str): Inference gather type: ``mean'' or ``max''.
        with_out_fc (bool): If True, add a final fully connected layer to the decoder.
        decoder_groups (int, optional): per level groups in the decoder.
        decoder_dropout (float): If specified, enables dropout with the given probability.
        coords_res (list of tuple of int, optional): list of inference resolutions for caching positional embedding.
    """
    def __init__(self, backbone, weight_mapper, in_nc=3, num_classes=3, kernel_sizes=3, level_layers=1, expand_ratio=1,
                 groups=1, inference_hflip=False, inference_gather='mean', with_out_fc=False, decoder_dropout=None):
        super(HyperGen, self).__init__()
        self.inference_hflip = inference_hflip
        self.inference_gather = inference_gather

        self.backbone = backbone()
        feat_channels = [in_nc] + self.backbone.feat_channels[:-1]
        self.decoder = MultiScaleDecoder(feat_channels, 3, num_classes, kernel_sizes, level_layers,
                                         with_out_fc=with_out_fc, out_kernel_size=1, expand_ratio=expand_ratio,
                                         dropout=decoder_dropout)
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
    def __init__(self, feat_channels, in_nc=3, num_classes=3, kernel_sizes=3, level_layers=1, norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6(inplace=True), out_kernel_size=1, expand_ratio=1, with_out_fc=False, dropout=None):
        super(MultiScaleDecoder, self).__init__()
        if isinstance(kernel_sizes, numbers.Number):
            kernel_sizes = (kernel_sizes,) * len(feat_channels)
        if isinstance(level_layers, numbers.Number):
            level_layers = (level_layers,) * len(feat_channels)
        assert len(kernel_sizes) == len(feat_channels), \
            f'kernel_sizes ({len(kernel_sizes)}) must be of size {len(feat_channels)}'
        assert len(level_layers) == len(feat_channels), \
            f'level_layers ({len(level_layers)}) must be of size {len(feat_channels)}'
        self.level_layers = level_layers
        self.levels = len(level_layers)
        self.layer_params = []
        feat_channels = feat_channels[::-1]  # Reverse the order of the feature channels

        # For each level
        prev_channels = 0
        for level in range(self.levels):
            curr_ngf = feat_channels[level]
            prev_channels += curr_ngf  # Accommodate the previous number of channels
            curr_layers = []
            kernel_size = kernel_sizes[level]

            # For each layer in the current level
            for layer in range(self.level_layers[level]):
                if (not with_out_fc) and (level == (self.levels - 1) and (layer == (self.level_layers[level] - 1))):
                    curr_ngf = num_classes
                if kernel_size > 1:
                    curr_layers.append(HyperPatchInvertedResidual(
                        prev_channels + 2, curr_ngf, kernel_size, expand_ratio=expand_ratio, norm_layer=norm_layer,
                        act_layer=act_layer))
                else:
                    curr_layers.append(make_meta_patch_conv2d_block(prev_channels + 2, curr_ngf, kernel_size))
                prev_channels = curr_ngf

            # Add level layers to module
            self.add_module(f'level_{level}', MetaSequential(*curr_layers))

        # Add the last layer
        if with_out_fc:
            out_fc_layers = [nn.Dropout2d(dropout, True)] if dropout is not None else []
            out_fc_layers.append(
                MetaPatchConv2d(prev_channels, num_classes, out_kernel_size, padding=out_kernel_size // 2))
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

    def forward(self, x, w):
        assert isinstance(w, (list, tuple))
        assert len(x) <= self.levels

        # For each level
        p = None
        for level in range(len(x)):
            level_w = w[level]
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
            p = torch.cat([get_image_coordinates(p.shape[0], *p.shape[-2:], p.device), p], dim=1)

            # Computer the output for the current level
            p = level_layers(p, level_w)

        # Last layer
        if self.out_fc is not None:
            p = self.out_fc(p, w[-1])

        return p


class HyperPatchInvertedResidual(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, expand_ratio=1, norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6(inplace=True), padding_mode='reflect'):
        super(HyperPatchInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_nc * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_nc == out_nc

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(make_meta_patch_conv2d_block(in_nc, hidden_dim, 1, norm_layer=norm_layer,
                                                       act_layer=act_layer))
        layers.extend([
            # dw
            make_meta_patch_conv2d_block(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim,
                                         norm_layer=norm_layer, act_layer=act_layer, padding_mode=padding_mode),
            # pw-linear
            make_meta_patch_conv2d_block(hidden_dim, out_nc, 1, stride=stride, norm_layer=norm_layer, act_layer=None)
        ])
        self.conv = MetaSequential(*layers)

    @property
    def hyper_params(self):
        return self.conv.hyper_params

    def forward(self, x, w):
        if self.use_res_connect:
            return x + self.conv(x, w)
        else:
            return self.conv(x, w)


def get_image_coordinates(b, h, w, device):
    x = torch.linspace(-1, 1, steps=w, device=device)
    y = torch.linspace(-1, 1, steps=h, device=device)
    grid = torch.stack(torch.meshgrid(y, x)[::-1], dim=0).repeat(b, 1, 1, 1)
    # grid = torch.stack(torch.meshgrid(x, y)[::-1], dim=0).repeat(b, 1, 1, 1)

    return grid


class WeightMapper(nn.Module):
    """ Weight mapper module (called context head in the paper).

    Args:
        in_channels (int): input number of channels.
        out_channels (int): output number of channels.
        levels (int): number of levels operating on different strides.
        bias (bool): if True, enables bias in all convolution operations.
        min_unit (int): used when dividing the signal channels into parts, must be a multiple of this number.
        weight_groups (int): per level signal to weights groups.
    """
    def __init__(self, in_channels, out_channels, levels=2, bias=False, min_unit=8, down_groups=1, flat_groups=1,
                 weight_groups=1, avg_pool=False):
        super(WeightMapper, self).__init__()
        assert levels > 0, 'levels must be greater than zero'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        self.bias = bias
        self.avg_pool = avg_pool
        self.down_groups = down_groups
        self.flat_groups = flat_groups
        self.weight_groups = weight_groups

        min_unit = max(min_unit, weight_groups)

        for level in range(self.levels - 1):
            down = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=bias, groups=down_groups),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True))
            self.add_module(f'down_{level}', down)

            up = nn.UpsamplingNearest2d(scale_factor=2)
            self.add_module(f'up_{level}', up)

            flat = [nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=bias, groups=flat_groups),
                    nn.BatchNorm2d(in_channels)]
            if level > 0:
                flat.append(nn.ReLU(inplace=True))
            flat = nn.Sequential(*flat)
            self.add_module(f'flat_{level}', flat)

        out_channels = [next_multiply(c, weight_groups) for c in out_channels]
        self.out_conv = Conv2dMulti(in_channels, out_channels, 1, bias=bias, min_unit=min_unit, groups=weight_groups)

    def forward(self, x):
        if self.levels <= 1:
            return self.out_conv(x)

        # Down stream
        feat = [x]
        for level in range(self.levels - 1):
            down = getattr(self, f'down_{level}')
            feat.append(down(feat[-1]))

        # Average the last feature map
        if self.avg_pool:
            orig_shape = feat[-1].shape
            if orig_shape[-2:] != (1, 1):
                feat[-1] = F.adaptive_avg_pool2d(feat[-1], 1)
                feat[-1] = F.interpolate(feat[-1], orig_shape[-2:], mode='nearest')

        # Up stream
        for level in range(self.levels - 2, -1, -1):
            up = getattr(self, f'up_{level}')
            flat = getattr(self, f'flat_{level}')

            x = up(feat.pop(-1))
            feat[-1] = torch.cat((feat[-1], x), dim=1)
            feat[-1] = flat(feat[-1])

        # Output weights
        w = self.out_conv(feat[-1])
        if self.weight_groups > 1:
            w = [wi[:, :oc] for wi, oc in zip(w, self.out_channels)]

        return w

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias}'


def next_multiply(x, base):
    return type(x)(np.ceil(x / base) * base)


class Conv2dMulti(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', min_unit=8):
        super(Conv2dMulti, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self._ranges = [0]

        in_nc_parts = divide_feature_legacy(in_channels, out_channels, min_unit)
        for i, out_nc in enumerate(out_channels):
            in_nc = in_nc_parts[i]
            self._ranges.append(self._ranges[-1] + in_nc)

            conv = nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
            self.add_module(f'conv_{i}', conv)

    def forward(self, x):
        out = []
        for i in range(len(self.out_channels)):
            linear = getattr(self, f'conv_{i}')
            out.append(linear(x[:, self._ranges[i]:self._ranges[i + 1]]))

        return out

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias}'


# Contains bugs (but used in some modules)
def divide_feature_legacy(in_feature, out_features, min_unit=8):
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
    remaining_units = units
    out_group_units = []
    for i, out_feat_group in enumerate(out_feat_groups):  # out_feat_group: (out_feature, indices array)
        if i < (len(out_feat_groups) - 1):
            curr_out_feat_size = out_feat_group[0] * len(out_feat_group[1])
            curr_units = max(curr_out_feat_size * units_feat_ratio, 1)
            curr_units = curr_units // len(out_feat_group[1]) * len(out_feat_group[1])  # Make divisible by num elements
            out_group_units.append(curr_units)
            remaining_units -= curr_units
        else:
            out_group_units.append(remaining_units)

    # Final feature division
    divided_in_features = np.zeros(len(out_features), dtype=int)
    for i, out_feat_group in enumerate(out_feat_groups):
        for j in range(len(out_feat_group[1])):
            divided_in_features[out_feat_group[1][j]] = out_group_units[i] // len(out_feat_group[1]) * min_unit

    return divided_in_features


def hyperseg_efficientnet(model_name, pretrained=False, levels=3, down_groups=1, flat_groups=1, weight_groups=1,
                          avg_pool=True, weights_path=None, **kwargs):
    from hyperseg.models.backbones.efficientnet import efficientnet
    from functools import partial

    weight_mapper = partial(WeightMapper, levels=levels, down_groups=down_groups, flat_groups=flat_groups,
                            weight_groups=weight_groups, avg_pool=avg_pool)
    backbone = partial(efficientnet, model_name, pretrained=pretrained, head=None, return_features=True)
    model = HyperGen(backbone, weight_mapper, **kwargs)

    if weights_path is not None:
        checkpoint = torch.load(weights_path)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=True)

    return model


def main(model='hyperseg.models.hyperseg_v0_1.hyperseg_efficientnet', res=(512,), pyramids=None,
         train=False):
    from hyperseg.utils.obj_factory import obj_factory
    from hyperseg.utils.utils import set_device
    from hyperseg.utils.img_utils import create_pyramid

    assert len(res) <= 2, f'res must be either a single number or a pair of numbers: "{res}"'
    res = res * 2 if len(res) == 1 else res

    device, gpus = set_device()
    model = obj_factory(model).to(device).train(train)
    x = torch.rand(2, 3, *res).to(device)
    x = create_pyramid(x, pyramids) if pyramids is not None else x
    pred = model(x)
    print(pred.shape)


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser('hyperseg test')
    parser.add_argument('-m', '--model', default='hyperseg.models.hyperseg_v0_1.hyperseg_efficientnet',
                        help='model object')
    parser.add_argument('-r', '--res', default=(512,), type=int, nargs='+',
                        metavar='N', help='image resolution')
    parser.add_argument('-p', '--pyramids', type=int, metavar='N',
                        help='number of image pyramids')
    parser.add_argument('-t', '--train', action='store_true',
                        help='If True, sets the model to training mode')
    main(**vars(parser.parse_args()))
