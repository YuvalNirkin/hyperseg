import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperseg.models.layers.meta_conv import MetaSequential


class MetaLinear(nn.Module):
    r"""Applies a dynamic linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=False):
        super(MetaLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.hyper_params = out_features * in_features

    def forward(self, x, w):
        """ Linear forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            w (torch.Tensor): Dynamic convolution weights.

        Returns:
            torch.Tensor: Dynamic linear transformation result.
        """
        assert x.shape[0] == w.shape[0]
        w = w.view(-1, self.out_features, self.in_features)
        return torch.bmm(w, x.unsqueeze(-1)).squeeze(-1)  # (b x o x i) @ (b x i x 1) -> (b x o x 1)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}'


def make_meta_linear_block(in_features, out_features, bias=False, norm_layer=nn.BatchNorm1d, act_layer=nn.ReLU(True),
                           dropout=None):
    """ Defines a Hyper convolution block with a normalization layer, an activation layer, and an optional
    dropout layer.

    Args:
        in_features (int): Input number of channels
        out_features (int): Output number of channels
        norm_layer (nn.Module): Type of feature normalization layer
        act_layer (nn.Module): Type of activation layer
        dropout (float): If specified, enables dropout with the given probability
    """
    assert dropout is None or isinstance(dropout, float)
    layers = [MetaLinear(in_features, out_features, bias)]
    if norm_layer is not None:
        layers.append(norm_layer(out_features))
    if act_layer is not None:
        layers.append(act_layer)
    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    return MetaSequential(*layers)


def main(model='hyperseg.models.layers.meta_linear.MetaLinear', in_features=3, out_features=5):
    from hyperseg.utils.obj_factory import obj_factory
    from hyperseg.utils.utils import set_device

    device, gpus = set_device()
    model = obj_factory(model, in_features=in_features, out_features=out_features).to(device)
    print(model)
    x = torch.rand(2, in_features).to(device)
    w = torch.ones(2, out_features * in_features).to(device)
    out = model(x, w)
    print(out.shape)


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='hyperseg.models.layers.meta_linear.MetaLinear',
                        help='model object')
    parser.add_argument('-if', '--in_features', default=3, type=int,
                        metavar='N', help='number of input features')
    parser.add_argument('-of', '--out_features', default=5, type=int,
                        metavar='N', help='number of output features')
    main(**vars(parser.parse_args()))
