import random
from enum import Enum
from PIL import Image, ImageFilter
from collections.abc import Sequence, Iterable
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.nn.functional import pad, interpolate
import torchvision.transforms as transforms
# from torchvision.transforms.transforms import _pil_interpolation_to_str


# Borrowed from: https://github.com/pytorch/vision/blob/v0.9.1/torchvision/transforms/functional.py
class InterpolationMode(Enum):
    """Interpolation modes
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


# Borrowed from: https://github.com/pytorch/vision/blob/v0.9.1/torchvision/transforms/functional.py
def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


def call_recursive(f, x):
    return [call_recursive(f, y) for y in x] if isinstance(x, (list, tuple)) else f(x)


class SegTransform(object):
    pass


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> img_landmarks_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3) or bounding box (4)
        Returns:
            Tensor or list of Tensor: Transformed images or poses
        """
        assert len(args) == 2 or (isinstance(args[0], (list, tuple)) and len(args[0]) == 2), \
            'Two arguments must be specified, an image and a corresponding label'
        input = list(args) if len(args) > 1 else args[0]
        for t in self.transforms:
            if isinstance(t, SegTransform):
                input = list(t(*input))
            else:
                input[0] = call_recursive(t, input[0])

        return tuple(input)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(SegTransform):
    """ Convert an image and pose in numpy.ndarray format to Tensor.

    Convert a numpy.ndarray image (H x W x C) in the range [0, 255] and numpy.ndarray pose (3)
    to torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] and torch.FloatTensor of shape (3)
    correspondingly.
    """

    def __call__(self, img, lbl):
        """
        Args:
            x (numpy.ndarray or list of numpy.ndarray): Image (H x W x C) or pose (3)

        Returns:
            numpy.ndarray or list of numpy.ndarray: Transformed images or poses
        """
        return call_recursive(F.to_tensor, img), torch.from_numpy(np.array(lbl).astype('long'))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=False):
        super(Normalize, self).__init__(mean, std, inplace)

    def __call__(self, x):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(x, self.mean, self.std, self.inplace)


def larger_edge_resize(img: Image, size, interpolation=Image.BICUBIC):
    """ Resize the given image to the target size. If size is a single number, the larger edge will be resized
    to that scale maintaining the image's aspect ratio.

    Args:
        img (PIL.Image): Input image.
        size (int or list of int): Target size
        interpolation (int): Interpolation type: Image.NEAREST, Image.BILINEAR, or Image.BICUBIC.

    Returns:

    """
    if not F._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w >= h and w == size) or (h >= w and h == size):
            return img
        if w < h:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
        else:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


class LargerEdgeResize(SegTransform, transforms.Resize):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            larger edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * width / height)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BICUBIC``
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        super(LargerEdgeResize, self).__init__(size, interpolation)

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img = larger_edge_resize(img, self.size, self.interpolation)
        lbl = larger_edge_resize(lbl, self.size, Image.NEAREST)

        return img, lbl

    def __repr__(self):
        interpolate_str = _interpolation_modes_from_int[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ConstantPad(SegTransform, transforms.Pad):
    """Pad the given PIL Image from the right and bottom to a constant resolution.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad the right and bottom borders. If tuple of length 2 is provided
            this is the padding on right and bottom respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, lbl_fill=None, padding_mode='constant'):
        super(ConstantPad, self).__init__(padding, fill, padding_mode)
        self.lbl_fill = fill if lbl_fill is None else lbl_fill

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be padded.
        Returns:
            PIL Image: Padded image.
        """
        padding = (0, 0) + tuple(np.maximum(self.padding - np.array(img.size), 0))  # left, top, right and bottom
        img = F.pad(img, padding, self.fill, self.padding_mode)
        lbl = F.pad(lbl, padding, self.lbl_fill, self.padding_mode)

        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, lbl_fill={2}, padding_mode={3})'. \
            format(self.padding, self.fill, self.lbl_fill, self.padding_mode)


class RandomResize(SegTransform):
    def __init__(self, p=0.5, scale_range=None, scale_values=None, interpolation=Image.BICUBIC):
        assert (scale_range is None) ^ (scale_values is None)
        self.p = p
        self.scale_range = scale_range
        self.scale_values = scale_values
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        if random.random() >= self.p:
            return img, lbl
        if self.scale_range is not None:
            scale = random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        else:
            scale = self.scale_values[random.randint(0, len(self.scale_values))]

        size = tuple(np.round(np.array(img.size[::-1]) * scale).astype(int))
        img = F.resize(img, size, self.interpolation)
        lbl = F.resize(lbl, size, Image.NEAREST)

        return img, lbl


class RandomCrop(SegTransform, transforms.RandomCrop):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, lbl_fill=None, padding_mode='constant'):
        super(RandomCrop, self).__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.lbl_fill = fill if lbl_fill is None else lbl_fill

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.

        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            lbl = F.pad(lbl, self.padding, self.lbl_fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            lbl = F.pad(lbl, (self.size[1] - lbl.size[0], 0), self.lbl_fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            lbl = F.pad(lbl, (0, self.size[0] - lbl.size[1]), self.lbl_fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)


class RandomHorizontalFlip(SegTransform):
    """Horizontally flip the given image and its corresponding label randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = F.hflip(img)
            lbl = F.hflip(lbl)

        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(SegTransform):
    """Vertically flip the given image and its corresponding label randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = F.vflip(img)
            lbl = F.vflip(lbl)

        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomGaussianBlur(object):
    """Apply gaussian blur randomly with a given probability.

    Args:
        p (float): probability of the image being blurred. Default value is 0.5
        r (int): gaussian filter radius. Default value is 5
    """

    def __init__(self, p=0.5, r=5):
        self.p = p
        self.r = r
        self.filter = ImageFilter.GaussianBlur(radius=r)

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(self.filter)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, r={})'.format(self.p, self.r)


class RandomRotation(SegTransform, transforms.RandomRotation):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=Image.BICUBIC, expand=False, center=None, fill=None, lbl_fill=None):
        super(RandomRotation, self).__init__(degrees, resample, expand, center, fill)
        self.lbl_fill = fill if lbl_fill is None else lbl_fill

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = random.uniform(self.degrees[0], self.degrees[1])
        img = F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)
        lbl = F.rotate(lbl, angle, Image.NEAREST, self.expand, self.center, self.fill)

        return img, lbl


class Pyramids(object):
    """Generate pyramids from an image.

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
    """

    def __init__(self, levels=1):
        assert levels >= 1
        self.levels = levels

    def __call__(self, img) -> list:
        """
        Args:
            img (PIL Image): Image to create the pyramids from.

        Returns:
            list of PIL Image: Image pyramids.
        """
        img_pyd = [img]
        for i in range(self.levels - 1):
            img_pyd.append(Image.fromarray(cv2.pyrDown(np.array(img_pyd[-1]))))

        return img_pyd

    def __repr__(self):
        return self.__class__.__name__ + '(levels={})'.format(self.levels)


class UpDownPyramids(Pyramids):
    """Generate pyramids from an image including upsampled pyramids.

    Args:
        levels (int): number of pyramid levels (must be 1 or greater)
        up_levels (int): number of upsampled pyramid levels (must be 0 or greater)
    """

    def __init__(self, levels=1, up_levels=0):
        super(UpDownPyramids, self).__init__(levels)
        assert up_levels >= 0
        self.up_levels = up_levels

    def __call__(self, img) -> list:
        """
        Args:
            img (PIL Image): Image to create the pyramids from.

        Returns:
            list of PIL Image: Image pyramids.
        """
        img_pyd = super(UpDownPyramids, self).__call__(img)
        for i in range(self.up_levels):
            img_pyd.append(Image.fromarray(cv2.pyrUp(np.array(img_pyd[0]))))

        return img_pyd

    def __repr__(self):
        return self.__class__.__name__ + '(levels={}, up_levels={})'.format(self.levels, self.up_levels)


def main(input, label, img_transforms=None, tensor_transforms=None):
    from hyperseg.utils.obj_factory import obj_factory
    from hyperseg.utils.img_utils import tensor2rgb
    from hyperseg.datasets.seg_transforms import Compose
    from PIL import Image

    # Initialize transforms
    img_transforms = obj_factory(img_transforms) if img_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    transform = Compose(img_transforms + tensor_transforms)

    # Read input image and corresponding label
    img = Image.open(input).convert('RGB')
    lbl = Image.open(label)
    palette = lbl.getpalette()

    # Apply transformations
    img_t, lbl_t = transform(img, lbl)

    if isinstance(img_t, (list, tuple)):
        img_t = img_t[-1]
        if lbl_t.shape[-2:] != img_t.shape[-2:]:
            lbl_t = lbl_t
            lbl_t = interpolate(lbl_t.float().view(1, 1, *lbl_t.shape), img_t.shape[-2:],
                                mode='nearest').long().squeeze()

    # Render results
    img, lbl = np.array(img), np.array(lbl.convert('RGB'))
    img_t = img_t[0] if isinstance(img_t, (list, tuple)) else img_t
    img_t = tensor2rgb(img_t)
    lbl_t = Image.fromarray(lbl_t.squeeze().numpy().astype('uint8'), mode='P')
    lbl_t.putpalette(palette)
    lbl_t = np.array(lbl_t.convert('RGB'))

    render_img_orig = np.concatenate((img, lbl), axis=1)
    render_img_transformed = np.concatenate((img_t, lbl_t), axis=1)
    f, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].imshow(render_img_orig)
    ax[1].imshow(render_img_transformed)
    plt.show()
    pass


if __name__ == "__main__":
    # Parse program arguments
    import os
    import argparse

    parser = argparse.ArgumentParser(os.path.splitext(os.path.basename(__file__))[0])
    parser.add_argument('input', metavar='PATH',
                        help='path to input image')
    parser.add_argument('-l', '--label', metavar='PATH',
                        help='path to segmentation label')
    parser.add_argument('-it', '--img_transforms', nargs='+', help='Numpy transforms')
    parser.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms')
    main(**vars(parser.parse_args()))
