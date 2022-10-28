import os
import tarfile
from zipfile import ZipFile
import shutil
import collections
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from tqdm import tqdm


VOC_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
VOC_MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
SBD_URL = 'http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip'
SBD_SPLITS_URL = 'http://cs.jhu.edu/~cxliu/data/list.zip'
COLOR_MAP = np.array([(0, 0, 0),  # 0=background
                      # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                      (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                      # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                      (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                      # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                      (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                      # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                      (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])


class VOCSBDDataset(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        cache_image_classes (bool, optional): If true, cache the number of per-pixel classes in each image.
    """
    def __init__(self, root, pair_list, transform=None, target_transform=None, transforms=None,
                 cache_image_classes=True):
        super(VOCSBDDataset, self).__init__(root, transforms, transform, target_transform)
        # Build dataset if it doesn't already exists
        download_extract(root)

        # Parse pair list file
        voc_root = os.path.join(root, 'VOCdevkit', 'VOC2012')
        pair_list = pair_list if pair_list.endswith('.txt') else pair_list + '.txt'
        pair_list_path = pair_list if os.path.isfile(pair_list) else os.path.join(voc_root, pair_list)
        pair_rel_paths = np.loadtxt(pair_list_path, dtype=str)
        pair_abs_paths = np.core.defchararray.add(voc_root, pair_rel_paths)
        if pair_abs_paths.ndim > 1:
            self.images = pair_abs_paths[:, 0]
            self.masks = pair_abs_paths[:, 1]
        else:   # No available masks
            self.images = pair_abs_paths
            self.masks = None

        # Add additional necessary arguments
        self.classes = list(range(21))
        self.weights = np.ones(len(self.images))
        self.color_map = COLOR_MAP

        # Calculate classes per image
        self.image_classes = None
        if self.masks is not None:
            cache_file = os.path.splitext(pair_list_path)[0] + '.npy' if cache_image_classes else None
            self.image_classes = calc_classes_per_image(self.masks, 21, cache_file)
            self.weights = calc_weights_from_image_classes(self.image_classes)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        # Image.fromarray(np.zeros(img.size[::-1], 'uint8'))
        if self.masks is not None:
            target = Image.open(self.masks[index])
        else:
            target = Image.fromarray(np.zeros(img.size[::-1], 'uint8'))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, np.array(target).astype('int64') if self.masks is not None else index

    def __len__(self):
        return len(self.images)


def download_extract(root):
    # Pascal VOC
    voc_filename = os.path.split(VOC_URL)[1]
    voc_path = os.path.join(root, voc_filename)
    voc_dir_path = os.path.join(root, 'VOCdevkit', 'VOC2012')
    if not os.path.isdir(voc_dir_path):
        if not os.path.isfile(voc_path):
            download_url(VOC_URL, root, voc_filename, VOC_MD5)
        with tarfile.open(voc_path, "r") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=root)
        os.remove(voc_path)

    # Pascal SBD
    sbd_filename = os.path.split(SBD_URL)[1]
    sbd_path = os.path.join(root, sbd_filename)
    sbd_dir_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClassAug')
    if not os.path.isdir(sbd_dir_path):
        if not os.path.isfile(sbd_path):
            download_url(SBD_URL, root, sbd_filename)
        with ZipFile(sbd_path, 'r') as zip_obj:
            zip_obj.extractall(voc_dir_path)
        sbd_temp_dir_path = os.path.join(root, 'VOCdevkit', 'VOC2012', os.path.splitext(sbd_filename)[0])
        os.rename(sbd_temp_dir_path, sbd_dir_path)

    # Pascal SBD splits
    sbd_splits_filename = os.path.split(SBD_SPLITS_URL)[1]
    sbd_splits_path = os.path.join(root, sbd_splits_filename)
    sbd_train_list_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'train.txt')
    if not os.path.isfile(sbd_train_list_path):
        if not os.path.isfile(sbd_splits_path):
            download_url(SBD_SPLITS_URL, root, sbd_splits_filename)
        with ZipFile(sbd_splits_path, 'r') as zip_obj:
            zip_obj.extractall(voc_dir_path)
        sbd_splits_temp_dir_path = os.path.join(root, 'VOCdevkit', 'VOC2012', os.path.splitext(sbd_splits_filename)[0])
        for file in os.listdir(sbd_splits_temp_dir_path):
            shutil.move(os.path.join(sbd_splits_temp_dir_path, file), voc_dir_path)
        os.rmdir(sbd_splits_temp_dir_path)


def calc_classes_per_image(masks_list, num_classes, cache_file=None):
    if cache_file is not None and os.path.isfile(cache_file):
        return np.load(cache_file)
    else:
        image_classes = np.zeros((len(masks_list), num_classes))
        for i, mask_path in enumerate(tqdm(masks_list, unit='files')):
            mask = np.array(Image.open(mask_path))
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


def main(dataset='hyperseg.datasets.voc_sbd.VOCSBDDataset',
         train_img_transforms=None, val_img_transforms=None,
         tensor_transforms=('seg_transforms.ToTensor', 'seg_transforms.Normalize'),
         workers=4, batch_size=4):
    from hyperseg.utils.obj_factory import obj_factory

    dataset = obj_factory(dataset)
    print(len(dataset))


if __name__ == "__main__":
    # Parse program arguments
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', metavar='OBJ', default='hyperseg.datasets.voc_sbd.VOCSBDDataset',
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
