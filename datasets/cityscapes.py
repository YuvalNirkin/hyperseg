import os
import json
from collections import namedtuple
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str

# import torchvision.datasets.cityscapes as cityscapes


CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])


# Based on: https://github.com/pytorch/vision/blob/master/torchvision/datasets/cityscapes.py
class CityscapesDataset(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string or list, optional): The image splits to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        cache_image_classes (bool, optional): If true, cache the number of per-pixel classes in each image.
        use_train_labels (bool, optional): If true, the train labels will be used following `train_id`.
        return_indices (bool, optional): If true, the sample index will be returned as well.


    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='bar ilancoarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = np.array([c.color for c in classes if not c.ignore_in_eval] + [(0, 0, 0)])
    id_to_train_id = np.array([c.train_id for c in classes], dtype='uint8')

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None, target_transform=None,
                 transforms=None, cache_image_classes=True, use_train_labels=True, return_indices=None):
        super(CityscapesDataset, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.splits = split if isinstance(split, (list, tuple)) else [split]
        self.images_dirs = [os.path.join(self.root, 'leftImg8bit', split) for split in self.splits]
        self.targets_dirs = [os.path.join(self.root, self.mode, split) for split in self.splits]
        self.target_type = target_type
        self.images = []
        self.targets = []

        # Verification
        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        for split in self.splits:
            verify_str_arg(split, "split", valid_modes, msg.format(split, mode, iterable_to_str(valid_modes)))

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]

        for i in range(len(self.splits)):
            if not os.path.isdir(self.images_dirs[i]) or not os.path.isdir(self.targets_dirs[i]):

                if split == 'train_extra':
                    image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
                else:
                    image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

                if self.mode == 'gtFine':
                    target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
                elif self.mode == 'gtCoarse':
                    target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

                if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                    extract_archive(from_path=image_dir_zip, to_path=self.root)
                    extract_archive(from_path=target_dir_zip, to_path=self.root)
                else:
                    raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                       ' specified "split" and "mode" are inside the "root" directory')

        # Parse image directories
        for i in range(len(self.splits)):
            for city in os.listdir(self.images_dirs[i]):
                img_dir = os.path.join(self.images_dirs[i], city)
                target_dir = os.path.join(self.targets_dirs[i], city)
                for file_name in os.listdir(img_dir):
                    target_types = []
                    for t in self.target_type:
                        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                     self._get_target_suffix(self.mode, t))
                        target_types.append(os.path.join(target_dir, target_name))

                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(target_types)

        # Handle the case for using only the train labels
        self.classes = [c for c in CityscapesDataset.classes if not c.ignore_in_eval] \
            if use_train_labels else CityscapesDataset.classes
        self.use_train_labels = use_train_labels

        # Add additional necessary arguments
        self.weights = np.ones(len(self.images))

        # Calculate classes per image
        self.image_classes = None
        if 'semantic' in self.target_type and 'test' not in self.splits:
            cache_file = os.path.join(root, f'{"_".join(sorted(self.splits))}.npy') if cache_image_classes else None
            type_index = np.where(np.array(self.target_type) == 'semantic')[0][0]
            masks = [p[type_index] for p in self.targets]
            self.image_classes = self.calc_classes_per_image(masks, cache_file)
            self.weights = calc_weights_from_image_classes(self.image_classes)

        self.return_indices = self.splits[0] == 'test' if return_indices is None else return_indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
                if self.use_train_labels:
                    target = np.array(target)
                    target[np.bitwise_or(target < 0, target >= len(CityscapesDataset.id_to_train_id))] = 0
                    target = Image.fromarray(CityscapesDataset.id_to_train_id[target], mode='P')

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # return image, target
        # return image, np.array(target).astype('int64') if self.splits[0] != 'test' else index
        return image, np.array(target).astype('int64') if not self.return_indices else index

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

    @property
    def color_map(self):
        return [c.color for c in self.classes]
        # if self.use_train_labels:
        #     return [c.color for c in self.classes if not c.ignore_in_eval]
        # else:
        #     return [c.color for c in self.classes]

    def calc_classes_per_image(self, masks_list, cache_file=None):
        num_classes = len(self.classes)
        if cache_file is not None and os.path.isfile(cache_file):
            return np.load(cache_file)
        else:
            image_classes = np.zeros((len(masks_list), num_classes))
            for i, mask_path in enumerate(tqdm(masks_list, unit='files')):
                mask = np.array(Image.open(mask_path))
                if self.use_train_labels:
                    mask = CityscapesDataset.id_to_train_id[mask]
                image_classes[i] += (np.bincount(mask[mask < num_classes], minlength=num_classes) > 0)

            if cache_file is not None:
                np.save(cache_file, image_classes)

        return image_classes


def calc_weights_from_image_classes(image_classes):
    class_occurances = image_classes.sum(axis=0)
    class_weights = np.sum(class_occurances) / (class_occurances + 1e-6)
    weights = np.sum(image_classes * class_weights, axis=1)
    weights = weights / np.sum(weights)

    return weights


class TargetLabelMapping(object):
    def __init__(self, label_map: np.array):
        super(TargetLabelMapping, self).__init__()
        self.label_map = label_map

    def __call__(self, target):
        """
        Args:
            target (PIL Image): Target labels to map.
        Returns:
            PIL Image: Transformed image.
        """
        return Image.fromarray(self.label_map[np.array(target)])


def main(dataset='hyperseg.datasets.cityscapes.CityscapesDataset',
         train_img_transforms=None, val_img_transforms=None,
         tensor_transforms=('seg_transforms.ToTensor', 'seg_transforms.Normalize'),
         workers=4, batch_size=4):
    from hyperseg.utils.obj_factory import obj_factory

    dataset = obj_factory(dataset)
    data = dataset[0]
    print(len(dataset))


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='OBJ', default='hyperseg.datasets.cityscapes.CityscapesDataset',
                        help='dataset object')
    parser.add_argument('-tit', '--train_img_transforms', nargs='+',
                        help='train image transforms')
    parser.add_argument('-vit', '--val_img_transforms', nargs='+',
                        help='validation image transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                        default=('seg_transforms.ToTensor', 'seg_transforms.Normalize'))
    parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                        help='mini-batch size')
    main(**vars(parser.parse_args()))
