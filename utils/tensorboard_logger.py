from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TensorBoardLogger(SummaryWriter):
    def __init__(self, log_dir=None):
        super(TensorBoardLogger, self).__init__(log_dir)
        self.__tb_logger = SummaryWriter(log_dir) if log_dir is not None else None
        self.log_dict = {}

    def reset(self, prefix=None):
        self.prefix = prefix
        self.log_dict.clear()

    def update(self, category='losses', **kwargs):
        if category not in self.log_dict:
            self.log_dict[category] = {}
        category_dict = self.log_dict[category]
        for key, val in kwargs.items():
            if key not in category_dict:
                category_dict[key] = AverageMeter()
            category_dict[key].update(val)

    def log_scalars_val(self, main_tag, global_step=None, category=None):
        if self.__tb_logger is not None:
            if category is not None:
                val_dict = {k: v.val for k, v in self.log_dict[category].items()}
                self.__tb_logger.add_scalars(main_tag + '/' + category, val_dict, global_step)
            else:
                for category, category_dict in self.log_dict.items():
                    val_dict = {k: v.val for k, v in category_dict.items()}
                    self.__tb_logger.add_scalars(main_tag + '/' + category, val_dict, global_step)

    def log_scalars_avg(self, main_tag, global_step=None, category=None):
        if self.__tb_logger is not None:
            if category is not None:
                val_dict = {k: v.avg for k, v in self.log_dict[category].items()}
                self.__tb_logger.add_scalars(main_tag + '/' + category, val_dict, global_step)
            else:
                for category, category_dict in self.log_dict.items():
                    val_dict = {k: v.avg for k, v in category_dict.items()}
                    self.__tb_logger.add_scalars(main_tag + '/' + category, val_dict, global_step)

    def log_image(self, tag, img_tensor, global_step=None):
        if self.__tb_logger is not None:
            self.__tb_logger.add_image(tag, img_tensor, global_step)

    def log_heatmap(self, tag, heatmap_tensor, global_step=None, xlabel=None, scale=0.5):
        if self.__tb_logger is None:
            return
        assert heatmap_tensor.dim() == 2

        # Convert tensor to dataframe
        rows, cols = range(heatmap_tensor.shape[0]), range(heatmap_tensor.shape[1])
        heatmap_df = pd.DataFrame(heatmap_tensor.cpu().numpy(), rows, cols)

        # Create figure
        figsize = np.maximum(np.round(np.array(heatmap_tensor.shape[::-1]) * scale).astype(int), 1)
        figure = plt.figure(figsize=figsize)
        sns.heatmap(heatmap_df, annot=True, cbar=False, fmt='.02f')
        if xlabel is not None:
            plt.xlabel(xlabel)

        self.__tb_logger.add_figure(tag, figure, global_step)


    def __str__(self):
        desc = '' if self.prefix is None else self.prefix
        for category, category_dict in self.log_dict.items():
            desc += '{}: ['.format(category)
            for key, log in category_dict.items():
                desc += '{}: {:.4f} ({:.4f}); '.format(key, log.val, log.avg)
            desc += '] '

        return desc
