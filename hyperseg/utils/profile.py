from functools import partial
import logging
import numpy as np
from torch.nn.modules.conv import _ConvNd
import torch
import torch.nn as nn


def profile(model: nn.Module, inputs, custom_ops=None, verbose=True, max_depth=None):
    custom_ops = {} if custom_ops is None else custom_ops
    flops_summary = []
    params_summary = {}
    hooks = {}
    types_collection = set()

    # Add hooks
    for name, module in model.named_modules():
        m_type = type(module)

        hook_fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            hook_fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (hook_fn.__qualname__, m_type))
        elif m_type in register_hooks:
            hook_fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (hook_fn.__qualname__, m_type))
        else:
            # hook_fn = zero_ops    # Debug
            if m_type not in types_collection and verbose:
                prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if hook_fn is not None:
            hooks[name] = (
                module.register_forward_hook(partial(hook_wrapper, fn=hook_fn, summary=flops_summary, name=name)),
                module.register_forward_hook(partial(hook_wrapper, fn=count_parameters, summary=params_summary,
                                                     name=name)))
        types_collection.add(m_type)

    # Run model
    prev_training_status = model.training
    model.eval()
    with torch.no_grad():
        model(*inputs)
    model.train(prev_training_status)

    # Remove hooks
    for m, (flops_handler, params_handler) in hooks.items():
        flops_handler.remove()
        params_handler.remove()

    if max_depth is not None:
        flops_summary, params_summary = clip_summary_depth(flops_summary, params_summary, max_depth, model)

    # Add final summary row
    flops_final_row, params_final_row = clip_summary_depth(flops_summary, params_summary, 0, model)
    flops_summary += flops_final_row
    params_summary = {**params_summary, **params_final_row}

    return flops_summary, params_summary


def print_summary(flops_summary, params_summary):
    # Create table
    table = []
    for module_name, class_name, input_shape, output_shape, flops in flops_summary:
        layer_name = f'{module_name} ({class_name})'
        shape_mapping = f'{shape2str(input_shape)} -> {shape2str(output_shape)}'
        params = f'{params_summary[module_name]:,}'
        flops = f'{flops / 1e9:.2f}B'
        table.append((layer_name, shape_mapping, params, flops))
    table = np.array(table)
    titles = ['Layer (type)', 'Shape Mapping', 'Params', 'FLOPs']
    col_sizes = [max([len(s) for s in table[:, i]] + [len(titles[i])]) for i in range(table.shape[1])]
    table_width = sum(col_sizes) + 4

    # Print title
    print('-' * table_width)
    print(f'{titles[0]:^{col_sizes[0]}}  {titles[1]:^{col_sizes[1]}} '
          f'{titles[2]:^{col_sizes[2]}} {titles[3]:^{col_sizes[3]}}')
    print('=' * table_width)

    # Print table
    for i, (layer_name, shape_mapping, params, flops) in enumerate(table):
        if i == (len(table) - 1):
            print('=' * table_width)
        print(f'{layer_name:>{col_sizes[0]}}  {shape_mapping:>{col_sizes[1]}} '
              f'{params:>{col_sizes[2]}} {flops:>{col_sizes[3]}}')
    print('-' * table_width)


def clip_summary_depth(flops_summary, params_summary, max_depth=1, model=None):
    module_dict = dict(model.named_modules()) if model is not None else None
    flops_summary_depth = []
    params_summary_depth = {}
    curr_module_name = None
    curr_input_shape = None
    flops_count = 0

    # For each row in the flops summary
    for i, (module_name, class_name, input_shape, output_shape, flops) in enumerate(flops_summary):
        module_parts = module_name.split('.')
        next_module_name = '.'.join(module_parts[:max_depth])
        if i == (len(flops_summary) - 1) or (curr_module_name is not None and curr_module_name != next_module_name):
            # Save previous module
            curr_class_name = module2str(module_dict[curr_module_name]) if module_dict is not None else None
            flops_summary_depth.append((curr_module_name, curr_class_name, curr_input_shape, curr_output_shape,
                                        flops_count))
            if module_dict is not None:
                params_summary_depth[curr_module_name] = count_parameters(module_dict[curr_module_name])
            else:
                params_summary_depth[curr_module_name] = \
                    sum([v for k, v in params_summary.items()
                         if k == curr_module_name or k.startswith(curr_module_name + '.')])

            # Reset
            curr_input_shape = None
            flops_count = 0

        # Process current module
        curr_module_name = next_module_name
        curr_input_shape = input_shape if curr_input_shape is None else curr_input_shape
        curr_output_shape = output_shape
        flops_count += flops

    return flops_summary_depth, params_summary_depth



def shape2str(shape):
    return 'x'.join([str(i) for i in shape]) if shape is not None else ''


def prRed(skk): print("\033[91m{}\033[00m".format(skk))


def prGreen(skk): print("\033[92m{}\033[00m".format(skk))


def prYellow(skk): print("\033[93m{}\033[00m".format(skk))


def module2str(module):
    return str(module.__class__).split(".")[-1].split("'")[0]


def hook_wrapper(m, x, y, fn, summary, name):
    if isinstance(summary, list):
        input_shape = x[0].shape if isinstance(x[0], torch.Tensor) else None
        output_shape = y.shape if isinstance(y, torch.Tensor) else None
        summary.append((name, module2str(m), input_shape, output_shape, fn(m, x, y)))
    else:
        summary[name] = fn(m, x, y)


def count_parameters(m, x=None, y=None):
    total_params = 0
    for p in m.parameters():
        total_params += p.numel()

    return total_params


def zero_ops(m, x, y):
    return 0


def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * np.int64(m.in_channels // m.groups * kernel_ops + bias_ops)

    return total_ops


def count_convNd_ver2(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = torch.zeros((y.size()[:1] + y.size()[2:])).numel()
    # Cout x Cin x Kw x Kh
    kernel_ops = m.weight.nelement()
    if m.bias is not None:
        # Cout x 1
        kernel_ops += + m.bias.nelement()
    # x N x H x W x Cout x (Cin x Kw x Kh + bias)

    return output_size * kernel_ops


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    return total_ops


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    return nelements


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    return total_ops


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return total_ops


def count_adap_avgpool(m, x, y):
    kernel = torch.DoubleTensor([*(x[0].shape[2:])]) // torch.DoubleTensor([*(y.shape[2:])])
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return total_ops


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    return total_ops


# nn.Linear
def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()
    total_ops = total_mul * num_elements

    return total_ops


##########################################################################
# RNN hooks
##########################################################################


def _count_rnn_cell(input_size, hidden_size, bias=True):
    # h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})
    total_ops = hidden_size * (input_size + hidden_size) + hidden_size
    if bias:
        total_ops += hidden_size * 2

    return total_ops


def count_rnn_cell(m: nn.RNNCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_rnn_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    return total_ops


def _count_gru_cell(input_size, hidden_size, bias=True):
    total_ops = 0
    # r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
    # z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
    state_ops = (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 2

    # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
    total_ops += (hidden_size + input_size) * hidden_size + hidden_size
    if bias:
        total_ops += hidden_size * 2
    # r hadamard : r * (~)
    total_ops += hidden_size

    # h' = (1 - z) * n + z * h
    # hadamard hadamard add
    total_ops += hidden_size * 3

    return total_ops


def count_gru_cell(m: nn.GRUCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_gru_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    return total_ops


def _count_lstm_cell(input_size, hidden_size, bias=True):
    total_ops = 0

    # i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    # f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    # o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
    # g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
    state_ops = (input_size + hidden_size) * hidden_size + hidden_size
    if bias:
        state_ops += hidden_size * 2
    total_ops += state_ops * 4

    # c' = f * c + i * g \\
    # hadamard hadamard add
    total_ops += hidden_size * 3

    # h' = o * \tanh(c') \\
    total_ops += hidden_size

    return total_ops


def count_lstm_cell(m: nn.LSTMCell, x: torch.Tensor, y: torch.Tensor):
    total_ops = _count_lstm_cell(m.input_size, m.hidden_size, m.bias)

    batch_size = x[0].size(0)
    total_ops *= batch_size

    return total_ops


def count_rnn(m: nn.RNN, x: torch.Tensor, y: torch.Tensor):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_rnn_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_rnn_cell(hidden_size * 2, hidden_size,
                                         bias) * 2
        else:
            total_ops += _count_rnn_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    return total_ops


def count_gru(m: nn.GRU, x: torch.Tensor, y: torch.Tensor):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_gru_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_gru_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_gru_cell(hidden_size * 2, hidden_size,
                                         bias) * 2
        else:
            total_ops += _count_gru_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    return total_ops


def count_lstm(m: nn.LSTM, x: torch.Tensor, y: torch.Tensor):
    bias = m.bias
    input_size = m.input_size
    hidden_size = m.hidden_size
    num_layers = m.num_layers

    if m.batch_first:
        batch_size = x[0].size(0)
        num_steps = x[0].size(1)
    else:
        batch_size = x[0].size(1)
        num_steps = x[0].size(0)

    total_ops = 0
    if m.bidirectional:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias) * 2
    else:
        total_ops += _count_lstm_cell(input_size, hidden_size, bias)

    for i in range(num_layers - 1):
        if m.bidirectional:
            total_ops += _count_lstm_cell(hidden_size * 2, hidden_size,
                                          bias) * 2
        else:
            total_ops += _count_lstm_cell(hidden_size, hidden_size, bias)

    # time unroll
    total_ops *= num_steps
    # batch_size
    total_ops *= batch_size

    return total_ops


##########################################################################
# Efficientnet hooks
##########################################################################
from hyperseg.models.backbones.efficientnet_utils import Conv2dStaticSamePadding, MemoryEfficientSwish


def count_conv2d_static_same_padding(m: Conv2dStaticSamePadding, x: (torch.Tensor,), y: torch.Tensor):
    return count_convNd(m, (m.static_padding(x[0]),), y)


def count_sigmoid(m, x, y):
    x = x[0]
    elements = x.numel()
    total_ops = 4 * elements

    return total_ops


def count_swish(m, x, y):
    total_ops = count_sigmoid(m, x, y)
    total_ops += x[0].numel()

    return total_ops


register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.

    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,

    # RNN
    nn.RNNCell: count_rnn_cell,
    nn.GRUCell: count_gru_cell,
    nn.LSTMCell: count_lstm_cell,
    nn.RNN: count_rnn,
    nn.GRU: count_gru,
    nn.LSTM: count_lstm,

    # Efficientnet
    Conv2dStaticSamePadding: count_conv2d_static_same_padding,
    MemoryEfficientSwish: count_swish,
}


def main(model, res=(512,), pyramids=None, max_depth=None):
    from hyperseg.utils.obj_factory import obj_factory
    from hyperseg.utils.utils import set_device
    from hyperseg.utils.img_utils import create_pyramid

    assert len(res) <= 2, f'res must be either a single number or a pair of numbers: "{res}"'
    res = res * 2 if len(res) == 1 else res

    device, gpus = set_device()
    model = obj_factory(model).to(device)

    x = torch.rand(1, 3, *res).to(device)
    x = create_pyramid(x, pyramids) if pyramids is not None else x

    # Run profile
    flops_summary, params_summary = profile(model, inputs=(x,), max_depth=max_depth)
    print_summary(flops_summary, params_summary)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', metavar='OBJ',
                        help='model object')
    parser.add_argument('-r', '--res', default=(512,), type=int, nargs='+',
                        metavar='N', help='image resolution')
    parser.add_argument('-p', '--pyramids', type=int, metavar='N',
                        help='number of image pyramids')
    parser.add_argument('-md', '--max_depth', type=int, metavar='N',
                        help='maximum module depth to print')
    main(**vars(parser.parse_args()))
