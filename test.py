""" Script for testing semantic segmentation models. """

import os
import argparse
import sys
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import pad
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from hyperseg.datasets.seg_transforms import Compose
from hyperseg.utils.utils import set_device, load_model
from hyperseg.utils.obj_factory import obj_factory
from hyperseg.utils.seg_utils import blend_seg, ConfusionMatrix
from hyperseg.utils.utils import remove_data_parallel_from_state_dict
from hyperseg.utils.img_utils import make_grid, tensor2rgb


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
general = parser.add_argument_group('general')
general.add_argument('exp_dir', metavar='DIR',
                     help='path to experiment directory')
general.add_argument('-m', '--model', metavar='PATH',
                     help='absolute or relative path (to the experiment directory) to the evaluation model')
general.add_argument('--gpus', nargs='+', type=int, metavar='N',
                     help='list of gpu ids to use (default: all)')
parser.add_argument('--cpu_only', action='store_true',
                    help='force cpu only')
general.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                     help='number of data loading workers (default: 4)')
general.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                     help='mini-batch size')
general.add_argument('-a', '--arch', metavar='STR',
                     help='override the architecture of the model')
general.add_argument('-dw', '--display_worst', type=int, metavar='N',
                     help='number of worst predicted image to display')
general.add_argument('-db', '--display_best', type=int, metavar='N',
                     help='number of best predicted image to display')
general.add_argument('-ds', '--display_sources', nargs='+',
                     help='additional display sources')
general.add_argument('-dwi', '--display_with_input', action='store_true',
                     help='adds the input image to the display')
general.add_argument('-da', '--display_alpha', type=float, metavar='F', default=0.75,
                     help='controls the opacity of the predictions and ground truth in the display image')
general.add_argument('-dbi', '--display_background_index', default=0, type=int, metavar='N',
                     help='background index to ignore in label color display')
general.add_argument('-f', '--forced', action='store_true',
                     help='force the execution of the test loop even when cache file exists')

data = parser.add_argument_group('data')
data.add_argument('-td', '--test_dataset', default='torchvision.datasets.voc.VOCSegmentation',
                  help='train dataset object')
data.add_argument('-it', '--img_transforms', nargs='+',
                  help='image transforms')
data.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                  default=('seg_transforms.ToTensor',
                           'seg_transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])'))
d = parser.get_default


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
    model.load_state_dict(remove_data_parallel_from_state_dict(checkpoint['state_dict']))
    model.train(train)

    if return_checkpoint:
        return model, checkpoint
    else:
        return model


def main(
        # General arguments
        exp_dir, model=d('model'), gpus=d('gpus'), cpu_only=d('cpu_only'), workers=d('workers'),
        batch_size=d('batch_size'),
        arch=d('arch'), display_worst=d('display_worst'), display_best=d('display_best'),
        display_sources=d('display_sources'), display_with_input=d('display_with_input'),
        display_alpha=d('display_alpha'), display_background_index=d('display_background_index'),
        forced=d('forced'),

        # Data arguments
        test_dataset=d('test_dataset'), img_transforms=d('img_transforms'), tensor_transforms=d('tensor_transforms')
):
    # Validation
    assert os.path.isdir(exp_dir), f'exp_dir "{exp_dir}" must be a path to a directory'
    model = 'model_best.pth' if model is None else model
    model = os.path.join(exp_dir, model) if not os.path.isfile(model) else model
    assert os.path.isfile(model), f'model path "{model}" does not exist'

    # Initialize cache directory
    cache_dir = os.path.join(exp_dir, os.path.splitext(os.path.basename(__file__))[0])
    scores_path = os.path.join(cache_dir, 'scores.npz')
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize device
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    device, gpus = set_device(gpus, not cpu_only)

    # Load segmentation model
    model = load_model(model, 'segmentation', device, arch)

    # Support multiple GPUs
    if gpus and len(gpus) > 1:
        model = nn.DataParallel(model, gpus)

    # Initialize transforms
    img_transforms = obj_factory(img_transforms) if img_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    test_transforms = Compose(img_transforms + tensor_transforms)

    # Initialize dataset
    test_dataset = obj_factory(test_dataset, transforms=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True,
                             drop_last=False, shuffle=False)

    # Initialize metric
    num_classes = len(test_dataset.classes)
    confmat = ConfusionMatrix(num_classes=num_classes)

    if forced or not os.path.isfile(scores_path):
        # For each batch of frames in the input video
        ious = []
        for i, (input, target) in enumerate(tqdm(test_loader, unit='batches', file=sys.stdout)):
            # Prepare input
            if isinstance(input, (list, tuple)):
                for j in range(len(input)):
                    input[j] = input[j].to(device)
            else:
                input = input.to(device)
            target = target.to(device)

            # Execute model
            pred = model(input)
            if pred.shape[2:] != target.shape[1:]:  # Make sure the prediction and target are of the same resolution
                pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear')

            # Update confusion matrix
            confmat.update(target.flatten(), pred.argmax(1).flatten() if pred.dim() == 4 else pred.flatten())

            # Calculate IoU scores
            for b in range(target.shape[0]):
                ious.append(jaccard(target[b].unsqueeze(0), pred[b].unsqueeze(0), num_classes, 0).item())
        # Save metrics to file
        ious = np.array(ious)
        global_acc, class_acc, class_iou = confmat.compute()
        global_acc = global_acc.item()
        class_acc = class_acc.cpu().numpy()
        class_iou = class_iou.cpu().numpy()
        np.savez(scores_path, ious=ious, global_acc=global_acc, class_acc=class_acc, class_iou=class_iou)
    else:  # Load metrics from file
        scores_archive = np.load(scores_path)
        ious = scores_archive['ious']
        global_acc = scores_archive['global_acc']
        class_acc = scores_archive['class_acc']
        class_iou = scores_archive['class_iou']

    # Print results
    print(f'global_acc={global_acc}')
    print(f'class_acc={class_acc}')
    print(f'class_iou={class_iou}')
    print(f'mIoU={np.mean(class_iou)}')

    # Display edge predictions
    indices = np.argsort(ious)
    if display_worst:
        print('Displaying worst predictions...')
        display_subset(test_dataset, indices[:display_worst], model, device, batch_size, scale=0.5, alpha=display_alpha,
                       with_input=display_with_input, display_sources=display_sources,
                       ignore_index=display_background_index)
    if display_best:
        print('Displaying best predictions...')
        display_subset(test_dataset, indices[-display_best:], model, device, batch_size, scale=0.5, alpha=display_alpha,
                       with_input=display_with_input, display_sources=display_sources,
                       ignore_index=display_background_index)


def calc_conf_mat(target, pred, num_classes, ignore_index=None):
    mask = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        mask &= (target != ignore_index)
    inds = num_classes * target[mask].to(torch.int64) + pred[mask]

    return torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def jaccard(target, pred, num_classes, ignore_index=None, eps=1e-6):
    confmat = calc_conf_mat(target.flatten(), pred.argmax(1).flatten(), num_classes, ignore_index)
    inter = torch.diag(confmat)
    union = confmat.sum(1) + confmat.sum(0) - inter
    if ignore_index is not None and ignore_index < len(union):
        union[ignore_index] = 0
    score = inter / (union + eps)

    return torch.mean(score[union > 0])


def display_subset(dataset, indices, model, device, batch_size=16, scale=0.5, alpha=0.75, with_input=True, dpi=100,
                   display_sources=None, ignore_index=0):
    data_loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=1,
                             pin_memory=True, drop_last=False, shuffle=False)
    inputs, preds, targets = [], [], []
    for i, (input, target) in enumerate(tqdm(data_loader, unit='batches', file=sys.stdout)):
        # Prepare input
        if isinstance(input, (list, tuple)):
            for j in range(len(input)):
                input[j] = input[j].to(device)
        else:
            input = input.to(device)
        target = target.to(device)

        # Execute model
        pred = model(input)

        # Make sure the prediction and target are of the same resolution
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest').long().squeeze(1)

        # Append
        inputs.append(input[0].cpu() if isinstance(input, (list, tuple)) else input.cpu())
        preds.append(pred.cpu())
        targets.append(target.cpu())
    inputs = torch.cat(inputs, dim=0)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min()) * 2. - 1.

    # Load display sources
    source_images = []
    if display_sources is not None:
        for display_source in display_sources:
            img_paths = glob(os.path.join(display_source, '*.png'))
            assert len(img_paths) == len(dataset), 'all display sources must be directories with the same number' \
                                                   ' of images as the dataset'
            img_paths = np.array(img_paths)[indices]
            imgs = []
            size = inputs.shape[-2:][::-1]
            for img_path in img_paths:
                img = Image.open(img_path)
                padding = (0, 0) + tuple(np.maximum(size - np.array(img.size), 0))  # left, top, right and bottom
                img = pad(img, padding)
                imgs.append(img)
            imgs = [torch.from_numpy(np.array(img).astype('long')).unsqueeze(0) for img in imgs]
            imgs = torch.cat(imgs, dim=0).to(device)
            source_images.append(imgs)

    seg_sources = [blend_seg(inputs, src_img, dataset.color_map, alpha=alpha) for src_img in source_images]
    seg_pred = blend_seg(inputs, preds, dataset.color_map, alpha=alpha, ignore_index=ignore_index)
    seg_gt = blend_seg(inputs, targets, dataset.color_map, alpha=alpha, ignore_index=ignore_index)
    if with_input:
        grid = make_grid(inputs, *seg_sources, seg_pred, seg_gt, normalize=False, padding=0)
    else:
        grid = make_grid(*seg_sources, seg_pred, seg_gt, normalize=False, padding=0)
    grid = tensor2rgb(grid)

    fig_size = tuple((np.array(grid.shape[1::-1]) * scale // dpi).astype(int))
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
