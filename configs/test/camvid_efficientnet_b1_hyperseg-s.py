import os
import inspect
from functools import partial
from hyperseg.datasets.camvid import CamVidDataset
from torchvision.transforms import Resize
from hyperseg.test import main


if __name__ == '__main__':
    project_dir = os.path.dirname(inspect.getabsfile(main))
    exp_name = os.path.splitext(os.path.basename(__file__))[0]  # Make sure the config and model have the same base name
    exp_dir = os.path.join('tests', exp_name)
    model = os.path.join('weights', exp_name + '.pth')
    data_dir = 'data/camvid'    # Download from: https://www.kaggle.com/carlolepelaars/camvid#
    test_dataset = partial(CamVidDataset, data_dir, 'test')     # 960 x 720
    img_transforms = [Resize([576, 768])]

    os.chdir(project_dir)
    os.makedirs(exp_dir, exist_ok=True)
    main(exp_dir, model=model, test_dataset=test_dataset, img_transforms=img_transforms, forced=True)
