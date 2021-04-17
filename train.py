import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from hyperseg.utils.utils import set_device, set_seed, str2int, save_checkpoint, get_arch
from hyperseg.utils.img_utils import make_grid
from hyperseg.utils.obj_factory import obj_factory
from hyperseg.utils.seg_utils import blend_seg
from hyperseg.datasets.seg_transforms import Compose
from hyperseg.utils.tensorboard_logger import TensorBoardLogger


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
general = parser.add_argument_group('general')
general.add_argument('exp_dir', metavar='DIR',
                     help='path to experiment directory')
general.add_argument('-r', '--resume', metavar='PATH',
                     help='path to resume directory or checkpoint')
general.add_argument('-se', '--start-epoch', metavar='N',
                     help='manual epoch number (useful on restarts)')
general.add_argument('-e', '--epochs', default=90, type=int, metavar='N',
                     help='number of total epochs to run')
general.add_argument('-ti', '--train_iterations', metavar='N',
                     help='number of train iterations per resolution to run')
general.add_argument('-vi', '--val_iterations', metavar='N',
                     help='number of validation iterations per resolution to run')
general.add_argument('--gpus', nargs='+', type=int, metavar='N',
                     help='list of gpu ids to use (default: all)')
general.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                     help='number of data loading workers (default: 4)')
general.add_argument('-b', '--batch-size', default=(64,), type=int, metavar='N',
                     help='mini-batch size (default: 64)')
general.add_argument('--seed', type=int, metavar='N',
                     help='random seed')
general.add_argument('-lf', '--log_freq', default=20, type=int, metavar='N',
                     help='number of steps between each loss plot')
general.add_argument('-lmr', '--log_max_res', default=512, type=int, metavar='N',
                     help='maximum resolution of logged images (larger axis)')

data = parser.add_argument_group('data')
data.add_argument('-td', '--train_dataset', default='torchvision.datasets.voc.VOCSegmentation',
                  help='train dataset object')
data.add_argument('-vd', '--val_dataset',
                  help='val dataset object')
data.add_argument('-tit', '--train_img_transforms', nargs='+',
                  help='train image transforms')
data.add_argument('-vit', '--val_img_transforms', nargs='+',
                  help='validation image transforms')
data.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                  default=('seg_transforms.ToTensor', 'seg_transforms.Normalize'))

training = parser.add_argument_group('training')
training.add_argument('-o', '--optimizer', default='optim.Adam(betas=(0.5,0.999))',
                      help='network\'s optimizer object')
training.add_argument('-s', '--scheduler', default='lr_scheduler.StepLR(step_size=10,gamma=0.6)',
                      help='scheduler object')
training.add_argument('-c', '--criterion', default='nn.CrossEntropyLoss(ignore_index=255)',
                      help='criterion object')
training.add_argument('-m', '--model', default='fsgan.models.simple_unet.UNet(n_classes=3,feature_scale=1)',
                      help='model object')
training.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                      help='use pre-trained model')
training.add_argument('-be', '--benchmark', default='hyperseg.utils.seg_utils.IOUBenchmark',
                      help='benchmark object')
training.add_argument('-bs', '--batch_scheduler', action='store_true',
                      help='if True, the learning rate will be scheduled after each batch iteration')
d = parser.get_default


def main(
    # General arguments
    exp_dir, resume=d('resume'), start_epoch=d('start_epoch'), epochs=d('epochs'),
    train_iterations=d('train_iterations'), val_iterations=d('val_iterations'), gpus=d('gpus'), workers=d('workers'),
    batch_size=d('batch_size'), seed=d('seed'), log_freq=d('log_freq'), log_max_res=d('log_max_res'),

    # Data arguments
    train_dataset=d('train_dataset'), val_dataset=d('val_dataset'), train_img_transforms=d('train_img_transforms'),
    val_img_transforms=d('val_img_transforms'), tensor_transforms=d('tensor_transforms'),

    # Training arguments
    optimizer=d('optimizer'), scheduler=d('scheduler'), criterion=d('criterion'), model=d('model'),
    pretrained=d('pretrained'), benchmark=d('benchmark'), hard_negative_mining=d('hard_negative_mining'),
    batch_scheduler=d('batch_scheduler')
):
    def proces_epoch(dataset_loader, train=True):
        stage = 'TRAINING' if train else 'VALIDATION'
        total_iter = len(dataset_loader) * dataset_loader.batch_size * epoch
        pbar = tqdm(dataset_loader, unit='batches')
        logger.reset()

        # Set networks training mode
        model.train(train)

        # For each batch
        for i, (input, target) in enumerate(pbar):
            # Set logger prefix
            logger.prefix = f'{stage}: Epoch: {epoch + 1} / {epochs}; LR: {scheduler.get_last_lr()[0]:.1e}; '

            # Prepare input
            with torch.no_grad():
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

            # Calculate loss
            loss_total = criterion(pred, target)

            # Benchmark
            running_metrics.update(target.cpu().numpy(), pred.argmax(1).cpu().numpy())

            if train:
                # Update generator weights
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                # Scheduler step
                if batch_scheduler:
                    scheduler.step()

            logger.update('losses', total=loss_total)
            metric_scores = {'iou': running_metrics.get_scores()[0]["Mean IoU : \t"]}
            logger.update('bench', **metric_scores)
            total_iter += dataset_loader.batch_size

            # Batch logs
            pbar.set_description(str(logger))
            if train and i % log_freq == 0:
                logger.log_scalars_val('batch', total_iter)

        # Epoch logs
        logger.log_scalars_avg('epoch/%s' % ('train' if train else 'val'), epoch, category='losses')
        logger.log_scalars_val('epoch/%s' % ('train' if train else 'val'), epoch, category='bench')
        if not train:
            # Log images
            input = input[0] if isinstance(input, (list, tuple)) else input
            input = limit_resolution(input, log_max_res, 'bilinear')
            pred = limit_resolution(pred, log_max_res, 'bilinear')
            target = limit_resolution(target.unsqueeze(1), log_max_res, 'nearest').squeeze(1)
            seg_pred = blend_seg(input, pred, train_dataset.color_map, alpha=0.75)
            seg_gt = blend_seg(input, target, train_dataset.color_map, alpha=0.75)
            grid = make_grid(input, seg_pred, seg_gt)
            logger.log_image('vis', grid, epoch)

        return logger.log_dict['losses']['total'].avg, logger.log_dict['bench']['iou'].val

    #################
    # Main pipeline #
    #################
    global_iterations = epochs * train_iterations

    # Initialize logger
    logger = TensorBoardLogger(log_dir=exp_dir)

    # Setup seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Setup device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize datasets
    train_img_transforms = obj_factory(train_img_transforms) if train_img_transforms is not None else []
    tensor_transforms = obj_factory(tensor_transforms) if tensor_transforms is not None else []
    train_transforms = Compose(train_img_transforms + tensor_transforms)
    train_dataset = obj_factory(train_dataset, transforms=train_transforms)
    if val_dataset is not None:
        val_img_transforms = obj_factory(val_img_transforms) if val_img_transforms is not None else []
        val_transforms = Compose(val_img_transforms + tensor_transforms)
        val_dataset = obj_factory(val_dataset, transforms=val_transforms)

    # Initialize loaders
    sampler = RandomSampler(train_dataset, True, train_iterations) if train_iterations is not None else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, sampler=sampler,
                              shuffle=sampler is None, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    # Setup Metrics
    running_metrics = runningScore(len(train_dataset.classes))

    # Create model
    arch = get_arch(model, num_classes=len(train_dataset.classes))
    model = obj_factory(model, num_classes=len(train_dataset.classes)).to(device)

    # Optimizer and scheduler
    optimizer = obj_factory(optimizer, model.parameters())
    scheduler = obj_factory(scheduler, optimizer)

    # Resume
    start_epoch = 0
    best_iou = 0.
    if resume is None:
        model_path, checkpoint_dir = os.path.join(exp_dir, 'model_latest.pth'), exp_dir
    elif os.path.isdir(resume):
        model_path, checkpoint_dir = os.path.join(resume, 'model_latest.pth'), resume
    else:  # resume is path to a checkpoint file
        model_path, checkpoint_dir = resume, os.path.split(resume)[0]
    if os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(checkpoint_dir))
        # model
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else start_epoch
        best_iou = checkpoint['best_iou'] if 'best_iou' in checkpoint else best_iou
        model.apply(init_weights)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))
        if not pretrained:
            print("=> randomly initializing networks...")
            model.apply(init_weights)

    # Lossess
    criterion = obj_factory(criterion).to(device)

    # Benchmark
    # benchmark = obj_factory(benchmark).to(device)

    # Support multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # For each epoch
    for epoch in range(start_epoch, epochs):
        # Training step
        epoch_loss, epoch_iou = proces_epoch(train_loader, train=True)

        # Validation step
        if val_loader is not None:
            with torch.no_grad():
                running_metrics.reset()
                epoch_loss, epoch_iou = proces_epoch(val_loader, train=False)
        running_metrics.reset()

        # Schedulers step (in PyTorch 1.1.0+ it must follow after the epoch training and validation steps)
        if not batch_scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        # Save models checkpoints
        is_best = epoch_iou > best_iou
        best_iou = max(epoch_iou, best_iou)
        save_checkpoint(exp_dir, 'model', {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_iou': best_iou,
            'arch': arch
        }, is_best)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


def limit_resolution(img, max_res=512, mode='bilinear'):
    img_res = max(img.shape[-2:])
    if img_res <= max_res:
        return img

    scale = max_res / img_res
    if mode == 'nearest':
        return F.interpolate(img.float(), scale_factor=scale, mode=mode).long()
    else:
        return F.interpolate(img, scale_factor=scale, mode=mode)


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
