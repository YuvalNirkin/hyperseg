from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperseg.utils.profile import register_hooks as base_register_hooks, shape2str, prRed, module2str, hook_wrapper, \
    count_parameters


def profile(model: nn.Module, inputs, custom_ops=None, verbose=True, max_depth=None):
    custom_ops = {} if custom_ops is None else custom_ops
    flops_summary = []
    params_summary = {}
    meta_params_summary = {}
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
                                                     name=name)),
                module.register_forward_hook(partial(hook_wrapper, fn=count_meta_parameters,
                                                     summary=meta_params_summary, name=name)))
        types_collection.add(m_type)

    # Run model
    prev_training_status = model.training
    model.eval()
    with torch.no_grad():
        model(*inputs)
    model.train(prev_training_status)

    # Remove hooks
    for m, (flops_handler, params_handler, meta_params_handler) in hooks.items():
        flops_handler.remove()
        params_handler.remove()
        meta_params_handler.remove()

    if max_depth is not None:
        flops_summary, params_summary, meta_params_summary = clip_summary_depth(flops_summary, params_summary,
                                                                                meta_params_summary, max_depth, model)

    # Add final summary row
    flops_final_row, params_final_row, meta_params_final_row = clip_summary_depth(flops_summary, params_summary,
                                                                                  meta_params_summary, 0, model)
    flops_summary += flops_final_row
    params_summary = {**params_summary, **params_final_row}
    meta_params_summary = {**meta_params_summary, **meta_params_final_row}

    return flops_summary, params_summary, meta_params_summary


def print_summary(flops_summary, params_summary, meta_params_summary):
    # Create table
    table = []
    for module_name, class_name, input_shape, output_shape, flops in flops_summary:
        layer_name = f'{module_name} ({class_name})'
        shape_mapping = f'{shape2str(input_shape)} -> {shape2str(output_shape)}'
        params = f'{params_summary[module_name]:,}'
        if meta_params_summary[module_name] > 0:
            params += f' ({meta_params_summary[module_name]:,})'
        flops = f'{flops / 1e9:.2f}B'
        table.append((layer_name, shape_mapping, params, flops))
    table = np.array(table)
    titles = ['Layer (type)', 'Shape Mapping', 'Params (meta)', 'FLOPs']
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


def clip_summary_depth(flops_summary, params_summary, meta_params_summary, max_depth=1, model=None):
    module_dict = dict(model.named_modules()) if model is not None else None
    flops_summary_depth = []
    params_summary_depth = {}
    meta_params_summary_depth = {}
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
                meta_params_summary_depth[curr_module_name] = count_meta_parameters(module_dict[curr_module_name])
            else:
                params_summary_depth[curr_module_name] = \
                    sum([v for k, v in params_summary.items()
                         if k == curr_module_name or k.startswith(curr_module_name + '.')])
                meta_params_summary_depth[curr_module_name] = \
                    sum([v for k, v in meta_params_summary.items()
                         if k == curr_module_name or k.startswith(curr_module_name + '.')])

            # Reset
            curr_input_shape = None
            flops_count = 0

        # Process current module
        curr_module_name = next_module_name
        curr_input_shape = input_shape if curr_input_shape is None else curr_input_shape
        curr_output_shape = output_shape
        flops_count += flops

    # Hack to solve final submodule bug
    if max_depth == 0:
        flops_summary_depth[0] = flops_summary_depth[0][:-1] + (flops_summary_depth[0][-1] + flops,)

    return flops_summary_depth, params_summary_depth, meta_params_summary_depth


def count_meta_parameters(m, x=None, y=None):
    return m.hyper_params if hasattr(m, 'hyper_params') else 0


##########################################################################
# Meta modules hooks
##########################################################################
from hyperseg.models.layers.meta_conv import MetaConv2d
from hyperseg.models.layers.meta_patch import MetaPatchConv2d


def count_MetaConv2d(m: MetaConv2d, x: (torch.Tensor,), y: torch.Tensor):
    kernel_ops = np.prod(m.kernel_size)     # kW x kH

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops)

    return total_ops


def count_MetaPatchConv2d(m: MetaPatchConv2d, x: (torch.Tensor,), y: torch.Tensor):
    kernel_ops = np.prod(m.kernel_size)     # kW x kH

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops)

    return total_ops


register_hooks = {
    MetaConv2d: count_MetaConv2d,
    MetaPatchConv2d: count_MetaPatchConv2d,

    **base_register_hooks
}


def main(model, res=(512,), pyramids=None, up_pyramid=False, max_depth=None):
    from hyperseg.utils.obj_factory import obj_factory
    from hyperseg.utils.utils import set_device
    from hyperseg.utils.img_utils import create_pyramid

    assert len(res) <= 2, f'res must be either a single number or a pair of numbers: "{res}"'
    res = res * 2 if len(res) == 1 else res

    device, gpus = set_device()
    model = obj_factory(model).to(device)

    x = torch.rand(1, 3, *res).to(device)
    x = create_pyramid(x, pyramids) if pyramids is not None else x
    if up_pyramid:
        x.append(F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=False))    # Upsample x2

    # Run profile
    flops_summary, params_summary, meta_params_summary = profile(model, inputs=(x,), max_depth=max_depth)
    print_summary(flops_summary, params_summary, meta_params_summary)


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
    parser.add_argument('-up', '--up_pyramid', action='store_true',
                        help='If True, creates a pyramid of 2x resolution scale')
    parser.add_argument('-md', '--max_depth', type=int, metavar='N',
                        help='maximum module depth to print')
    main(**vars(parser.parse_args()))
