import os
import importlib
from functools import partial


KNOWN_MODULES = {
    # datasets
    'opencv_video_seq_dataset': 'hyperseg.datasets.opencv_video_seq_dataset',
    'img_landmarks_transforms': 'hyperseg.datasets.img_landmarks_transforms',
    'seg_transforms': 'hyperseg.datasets.seg_transforms',
    'transforms': 'torchvision.transforms',

    # models
    'models': 'hyperseg.models',
    'mobilenet': 'hyperseg.models.mobilenet',
    'efficientnet': 'hyperseg.models.efficientnet',
    'efficientnet_custom': 'hyperseg.models.efficientnet_custom',
    'efficientnet_custom_03': 'hyperseg.models.efficientnet_custom_03',

    # Layers
    'weight_mapper': 'hyperseg.models.layers.weight_mapper',
    'weight_mapper_unet': 'hyperseg.models.layers.weight_mapper_unet',

    # Torch
    'nn': 'torch.nn',
    'optim': 'torch.optim',
    'lr_scheduler': 'torch.optim.lr_scheduler',
}


def extract_args(*args, **kwargs):
    return args, kwargs


def is_str_module(obj_exp):
    return isinstance(obj_exp, str) and '.' in obj_exp or ('(' in obj_exp and ')' in obj_exp)


def obj_factory(obj_exp, *args, **kwargs):
    """ Creates objects from strings or partial objects with additional provided arguments.

    In case a sequence is provided, all objects in the sequence will be created recursively.
    Objects that are not strings or partials be returned as they are.

    Args:
        obj_exp (str or partial): The object string expresion or partial to be converted into an object. Can also be
            a sequence of object expressions
        *args: Additional arguments to pass to the object
        **kwargs: Additional keyword arguments to pass to the object

    Returns:
        object or object list: Created object or list of recursively created objects
    """
    if isinstance(obj_exp, (list, tuple)):
        return [obj_factory(o, *args, **kwargs) for o in obj_exp]
    if isinstance(obj_exp, partial):
        return obj_exp(*args, **kwargs)
    if not isinstance(obj_exp, str):
        return obj_exp

    # Handle arguments
    if '(' in obj_exp and ')' in obj_exp:
        args_exp = obj_exp[obj_exp.find('('):]
        obj_args, obj_kwargs = eval('extract_args' + args_exp)

        # # Recursively evaluate string modules
        # obj_args = tuple([obj_factory(o) if is_str_module(o) else o for o in obj_args])
        # obj_kwargs = {k: obj_factory(v) if is_str_module(v) else v for k, v in obj_kwargs.items()}

        # Concatenate arguments
        args = obj_args + args
        kwargs.update(obj_kwargs)

        obj_exp = obj_exp[:obj_exp.find('(')]

    # From here we can assume that dots in the remaining of the expression
    # only separate between modules and classes
    module_name, class_name = os.path.splitext(obj_exp)
    class_name = class_name[1:]
    module = importlib.import_module(KNOWN_MODULES[module_name] if module_name in KNOWN_MODULES else module_name)
    module_class = getattr(module, class_name)
    class_instance = module_class(*args, **kwargs)

    return class_instance


def partial_obj_factory(obj_exp, *args, **kwargs):
    """ Creates objects from strings or partial objects with additional provided arguments.

    In case a sequence is provided, all objects in the sequence will be created recursively.
    Objects that are not strings or partials be returned as they are.

    Args:
        obj_exp (str or partial): The object string expresion or partial to be converted into an object. Can also be
            a sequence of object expressions
        *args: Additional arguments to pass to the object
        **kwargs: Additional keyword arguments to pass to the object

    Returns:
        object or object list: Created object or list of recursively created objects
    """
    if isinstance(obj_exp, (list, tuple)):
        return [partial_obj_factory(o, *args, **kwargs) for o in obj_exp]
    if isinstance(obj_exp, partial):
        return partial(obj_exp.func, *(obj_exp.args + args), **{**obj_exp.keywords, **kwargs})
    if not isinstance(obj_exp, str):
        return partial(obj_exp)

    # Handle arguments
    if '(' in obj_exp and ')' in obj_exp:
        args_exp = obj_exp[obj_exp.find('('):]
        obj_args, obj_kwargs = eval('extract_args' + args_exp)

        # Concatenate arguments
        args = obj_args + args
        kwargs.update(obj_kwargs)

        obj_exp = obj_exp[:obj_exp.find('(')]

    # From here we can assume that dots in the remaining of the expression
    # only separate between modules and classes
    module_name, class_name = os.path.splitext(obj_exp)
    class_name = class_name[1:]
    module = importlib.import_module(KNOWN_MODULES[module_name] if module_name in KNOWN_MODULES else module_name)
    module_class = getattr(module, class_name)

    return partial(module_class, *args, **kwargs)


def main(obj_exp):
    # obj = obj_factory(obj_exp)
    # print(obj)

    import inspect
    partial_obj = partial_obj_factory(obj_exp)
    print(f'is obj_exp a class = {inspect.isclass(partial_obj.func)}')
    print(partial_obj)


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser('utils test')
    parser.add_argument('obj_exp', help='object string')
    args = parser.parse_args()

    main(args.obj_exp)
