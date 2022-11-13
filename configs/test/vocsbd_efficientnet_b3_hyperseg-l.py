import os
import inspect
from functools import partial
from hyperseg.datasets.voc_sbd import VOCSBDDataset
from hyperseg.datasets.seg_transforms import ConstantPad, ToTensor, Normalize
from hyperseg.test import main


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(inspect.getabsfile(main)))
    exp_name = os.path.splitext(os.path.basename(__file__))[0]  # Make sure the config and model have the same base name
    exp_dir = os.path.join('tests', exp_name)
    model = os.path.join('weights', exp_name + '.pth')
    data_dir = 'data/vocsbd'    # The dataset will be downloaded automatically
    test_dataset = partial(VOCSBDDataset, data_dir, 'val')
    img_transforms = [ConstantPad(512, lbl_fill=255)]
    tensor_transforms = [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    os.chdir(project_dir)
    os.makedirs(exp_dir, exist_ok=True)
    main(exp_dir, model=model, test_dataset=test_dataset, img_transforms=img_transforms,
         tensor_transforms=tensor_transforms, forced=True)
