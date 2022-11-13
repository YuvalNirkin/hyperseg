import os
import inspect
from functools import partial
import torch.optim as optim
from hyperseg.utils.polylr import PolyLR
from hyperseg.datasets.cityscapes import CityscapesDataset
from hyperseg.datasets.seg_transforms import RandomHorizontalFlip, RandomResize, RandomCrop, ToTensor, Normalize
from torchvision.transforms import ColorJitter, Resize
from hyperseg.losses.bootstrapped_ce_loss import BootstrappedCrossEntropyLoss
from hyperseg.models.hyperseg_v1_0_unify import hyperseg_efficientnet
from hyperseg.train import main


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(inspect.getabsfile(main)))
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('checkpoints/cityscapes', exp_name)
    data_dir = 'data/cityscapes'    # Download from: https://www.cityscapes-dataset.com
    train_dataset = partial(CityscapesDataset, data_dir, 'train', 'fine', 'semantic')
    val_dataset = partial(CityscapesDataset, data_dir, 'val', 'fine', 'semantic')
    val_img_transforms = [Resize([768, 1536])]
    train_img_transforms = [RandomResize(scale_range=(0.375, 1.5)),
                            RandomCrop([768, 768], pad_if_needed=True, lbl_fill=255), RandomHorizontalFlip(),
                            ColorJitter(0.25, 0.25, 0.25, 0.25)]
    tensor_transforms = [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    epochs = 360
    train_iterations = 4000
    batch_size = 16
    workers = 16
    pretrained = True
    optimizer = partial(optim.Adam, lr=1e-3, betas=(0.5, 0.999))
    scheduler_iterations = epochs * train_iterations // batch_size
    scheduler = partial(PolyLR, power=0.9, max_epoch=scheduler_iterations)
    batch_scheduler = True
    criterion = BootstrappedCrossEntropyLoss(ignore_index=255)
    model = partial(hyperseg_efficientnet, 'efficientnet-b1', pretrained=pretrained, levels=2,
                    out_feat_scale=[1., 0.166, 0.2, 0.25, 0.4], kernel_sizes=[1, 1, 1, 3, 3],
                    level_channels=[32, 16, 8, 8, 8], expand_ratio=2, with_out_fc=False, decoder_dropout=None,
                    weight_groups=[32, 16, 8, 16, 4], decoder_groups=1, inference_hflip=True, unify_level=4,
                    coords_res=[(768, 768), (768, 1536)])

    os.chdir(project_dir)
    os.makedirs(exp_dir, exist_ok=True)
    main(exp_dir, train_dataset=train_dataset, val_dataset=val_dataset, train_img_transforms=train_img_transforms,
         val_img_transforms=val_img_transforms, tensor_transforms=tensor_transforms, epochs=epochs,
         train_iterations=train_iterations, batch_size=batch_size, workers=workers, optimizer=optimizer,
         scheduler=scheduler, pretrained=pretrained, model=model, criterion=criterion, batch_scheduler=batch_scheduler)
