import os
import inspect
from functools import partial
import torch.optim as optim
from hyperseg.utils.polylr import PolyLR
from hyperseg.datasets.voc_sbd import VOCSBDDataset
from hyperseg.datasets.seg_transforms import RandomHorizontalFlip, ConstantPad, RandomResize, RandomRotation, \
    ToTensor, Normalize
from torchvision.transforms import ColorJitter
from hyperseg.models.hyperseg_v0_1 import hyperseg_efficientnet
from hyperseg.train import main


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(inspect.getabsfile(main)))
    exp_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_dir = os.path.join('checkpoints/vocsbd', exp_name)
    data_dir = 'data/vocsbd'
    train_dataset = partial(VOCSBDDataset, data_dir, 'train_aug')
    val_dataset = partial(VOCSBDDataset, data_dir, 'val')
    val_img_transforms = [ConstantPad(512, lbl_fill=255)]
    train_img_transforms = [RandomHorizontalFlip(), ColorJitter(0.5, 0.5, 0.5, 0.5),
                            RandomResize(scale_range=(0.25, 0.9)), RandomRotation(30.)] + val_img_transforms
    tensor_transforms = [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    epochs = 160
    train_iterations = 20_000
    batch_size = 32
    workers = 16
    pretrained = True
    optimizer = partial(optim.Adam, lr=1e-4, betas=(0.5, 0.999))
    scheduler = partial(PolyLR, power=3., max_epoch=epochs)
    model = partial(hyperseg_efficientnet, 'efficientnet-b3', pretrained=pretrained, levels=3,
                    kernel_sizes=(1, 1, 3, 3, 3, 3), expand_ratio=2, inference_hflip=True, with_out_fc=False,
                    decoder_dropout=None, weight_groups=16)

    os.chdir(project_dir)
    os.makedirs(exp_dir, exist_ok=True)
    main(exp_dir, train_dataset=train_dataset, val_dataset=val_dataset, train_img_transforms=train_img_transforms,
         val_img_transforms=val_img_transforms, tensor_transforms=tensor_transforms, epochs=epochs,
         train_iterations=train_iterations, batch_size=batch_size, workers=workers, optimizer=optimizer,
         scheduler=scheduler, pretrained=pretrained, model=model)

    # os.system('sudo shutdown')
