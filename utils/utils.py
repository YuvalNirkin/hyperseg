import os
import shutil
from functools import partial
from collections import OrderedDict
import torch
import random
import torch.nn.init as init
import numpy as np
import ffmpeg
import warnings
from itertools import groupby
import torch.backends.cudnn as cudnn
from .obj_factory import obj_factory


def init_weights(m, init_type='normal', gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


def set_device(gpus=None, use_cuda=True):
    use_cuda = torch.cuda.is_available() if use_cuda else use_cuda
    if use_cuda:
        gpus = list(range(torch.cuda.device_count())) if not gpus else gpus
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')

    return device, gpus


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def save_checkpoint(exp_dir, base_name, state, is_best=False):
    """ Saves a model's checkpoint.
    :param exp_dir: Experiment directory to save the checkpoint into.
    :param base_name: The output file name will be <base_name>_latest.pth and optionally <base_name>_best.pth
    :param state: The model state to save.
    :param is_best: If True <base_name>_best.pth will be saved as well.
    """
    filename = os.path.join(exp_dir, base_name + '_latest.pth')
    if 'state_dict' in state:
        state['state_dict'] = remove_data_parallel_from_state_dict(state['state_dict'])
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(exp_dir, base_name + '_best.pth'))


def remove_data_parallel_from_state_dict(state_dict):
    out_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')    # Remove '.module' from key
        out_state_dict[new_k] = v

    return out_state_dict


mag_map = {'K': 3, 'M': 6, 'B': 9}


def str2int(s):
    if isinstance(s, (list, tuple)):
        return [str2int(o) for o in s]
    if not isinstance(s, str):
        return s
    return int(float(s[:-1]) * 10 ** mag_map[s[-1].upper()]) if s[-1].upper() in mag_map else int(s)


def get_arch(obj, *args, eval_partial=True, **kwargs):
    """ Extract the architecture (string representation) of an object given as a string or partial together
    with additional provided arguments.

    The returned architecture can be used to create the object using the obj_factory function.

    Args:
        obj (str or partial): The object string expresion or partial to be converted into an object
        *args: Additional arguments to pass to the object
        **kwargs: Additional keyword arguments to pass to the object

    Returns:
        arch (str): The object's architecture (string representation).
    """
    obj_args, obj_kwargs = [], {}
    if isinstance(obj, str):
        if '(' in obj and ')' in obj:
            arg_pos = obj.find('(')
            func = obj[:arg_pos]
            args_exp = obj[arg_pos:]
            obj_args, obj_kwargs = eval('extract_args' + args_exp)
        else:
            func = obj
    elif isinstance(obj, partial):
        func = obj.func.__module__ + '.' + obj.func.__name__
        obj_args, obj_kwargs = obj.args, obj.keywords
    else:
        return None

    # Concatenate arguments
    obj_args = obj_args + args
    obj_kwargs.update(kwargs)

    # Recursively evaluate arguments
    obj_args = [get_arch(o, eval_partial=False) if isinstance(o, partial) else o for o in obj_args]
    obj_kwargs = {k: get_arch(v, eval_partial=False) if isinstance(v, partial) else v for k, v in obj_kwargs.items()}

    if not eval_partial:
        obj_args.insert(0, func)
        func = 'functools.partial'

    # Convert object components to string representation
    args = ",".join(map(repr, obj_args))
    kwargs = ",".join("{}={!r}".format(k, v) for k, v in obj_kwargs.items())
    comma = ',' if args != '' and kwargs != '' else ''
    format_string = '{func}({args}{comma}{kwargs})'
    arch = format_string.format(func=func, args=args, comma=comma, kwargs=kwargs).replace(' ', '')

    return arch


def load_model(model_path, name='', device=None, arch=None, return_checkpoint=False, train=False):
    """ Load a model from checkpoint.

    This is a utility function that combines the model weights and architecture (string representation) to easily
    load any model without explicit knowledge of its class.

    Args:
        model_path (str): Path to the model's checkpoint (.pth)
        name (str): The name of the model (for printing and error management)
        device (torch.device): The device to load the model to
        arch (str): The model's architecture (string representation)
        return_checkpoint (bool): If True, the checkpoint will be returned as well
        train (bool): If True, the model will be set to train mode, else it will be set to test mode

    Returns:
        (nn.Module, dict (optional)): A tuple that contains:
            - model (nn.Module): The loaded model
            - checkpoint (dict, optional): The model's checkpoint (only if return_checkpoint is True)
    """
    assert model_path is not None, '%s model must be specified!' % name
    assert os.path.exists(model_path), 'Couldn\'t find %s model in path: %s' % (name, model_path)
    print('=> Loading %s model: "%s"...' % (name, os.path.basename(model_path)))
    checkpoint = torch.load(model_path)
    assert arch is not None or 'arch' in checkpoint, 'Couldn\'t determine %s model architecture!' % name
    arch = checkpoint['arch'] if arch is None else arch
    model = obj_factory(arch)
    if device is not None:
        model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.train(train)

    if return_checkpoint:
        return model, checkpoint
    else:
        return model


def random_pair(n, min_dist=1, index1=None):
    """ Return a random pair of integers in the range [0, n) with a minimum distance between them.

    Args:
        n (int): Determine the range size
        min_dist (int): The minimum distance between the random pair
        index1 (int, optional): If specified, this will determine the first integer

    Returns:
        (int, int): The random pair of integers.
    """
    r1 = random.randint(0, n - 1) if index1 is None else index1
    d_left = min(r1, min_dist)
    d_right = min(n - 1 - r1, min_dist)
    r2 = random.randint(0, n - 2 - d_left - d_right)
    r2 = r2 + d_left + 1 + d_right if r2 >= (r1 - d_left) else r2

    return r1, r2


def random_pair_range(a, b, min_dist=1, index1=None):
    """ Return a random pair of integers in the range [a, b] with a minimum distance between them.

    Args:
        a (int): The minimum number in the range
        b (int): The maximum number in the range
        min_dist (int): The minimum distance between the random pair
        index1 (int, optional): If specified, this will determine the first integer

    Returns:
        (int, int): The random pair of integers.
    """
    r1 = random.randint(a, b) if index1 is None else index1
    d_left = min(r1 - a, min_dist)
    d_right = min(b - r1, min_dist)
    r2 = random.randint(a, b - 1 - d_left - d_right)
    r2 = r2 + d_left + 1 + d_right if r2 >= (r1 - d_left) else r2

    return r1, r2


def get_media_info(media_path):
    """ Return media information.

    Args:
        media_path (str): Path to media file

    Returns:
        (int, int, int, float): Tuple containing:
            - width (int): Frame width
            - height (int): Frame height
            - total_frames (int): Total number of frames (will be 1 for images)
            - fps (float): Frames per second (irrelevant for images)
    """
    assert os.path.isfile(media_path), f'The media file does not exist: "{media_path}"'
    probe = ffmpeg.probe(media_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    total_frames = int(video_stream['nb_frames']) if 'nb_frames' in video_stream else 1
    fps_part1, fps_part2 = video_stream['r_frame_rate'].split(sep='/')
    fps = float(fps_part1) / float(fps_part2)

    return width, height, total_frames, fps


def get_media_resolution(media_path):
    return get_media_info(media_path)[:2]


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
    for i, out_feat_group in enumerate(out_feat_groups):    # out_feat_group: (out_feature, indices array)
        if i < (len(out_feat_groups) - 1):
            curr_out_feat_size = out_feat_group[0] * len(out_feat_group[1])
            curr_units = max(curr_out_feat_size * units_feat_ratio, 1)
            curr_units = curr_units // len(out_feat_group[1]) * len(out_feat_group[1])   # Make divisible by num elements
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
        elif len(out_feat_groups) == 1:
            out_group_units[-1] += remaining_units
        else:       # TODO: major bug! Remove this else completly
            out_group_units.append(remaining_units)

    # Final feature division
    divided_in_features = np.zeros(len(out_features), dtype=int)
    for i, out_feat_group in enumerate(out_feat_groups):
        for j in range(len(out_feat_group[1])):
            divided_in_features[out_feat_group[1][j]] = out_group_units[i] // len(out_feat_group[1]) * min_unit

    return divided_in_features


class ExpDecayingHyperParameter(object):
    def __init__(self, init_value, final_value, half_life):
        self.init_value = init_value
        self.final_value = final_value
        self.half_life = half_life
        self.iterations = 0

    def step(self):
        self.iterations += 1

    def state_dict(self):
        """Returns the state of the parameter as a :class:`dict`.

        It contains an entry for every variable in self.__dict__.
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the parameter's state.

        Arguments:
            state_dict (dict): parameter state. Should be an object returned from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def __call__(self):
        factor = 0.5 ** (self.iterations / self.half_life)
        return self.init_value * factor + self.final_value * (1. - factor)
