import io
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl

import PIL.Image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)))

    return new_cmap


class Logger:
    def __init__(self, model_fname, cfg):
        self.cfg = cfg
        self.model_fname = model_fname
        self.writer = SummaryWriter(log_dir='runs/' + model_fname)

    def log_config(self):
        self.writer.add_text('Info/Config', json.dumps(self.cfg), 0)

    def log_train(self, data, n_iter):
        self.writer.add_scalar('Loss/Train', data['loss_train'], n_iter)

    def log_eval(self, data, n_iter):
        self.writer.add_scalar('Loss/Eval', data['loss_eval'], n_iter)
        for k, v in data['Problem_Misclassifications'].items():
            self.writer.add_scalar(
                'Problem_Misclassifications/' + k,
                v,
                n_iter
            )
        self.writer.add_scalar(
            'Total_Misclassifications',
            data['Total_Misclassifications'],
            n_iter
        )

    def log_confusion_matrix(self, mc_table):
        cmap_mod = truncate_colormap('Greens', minval=.4, maxval=.99)
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(mc_table, cmap=cmap_mod, vmin=0, vmax=1)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Text annotations
        for i in range(mc_table.shape[0]):
            for j in range(mc_table.shape[0]):
                ax.text(j, i, '%.02f' % mc_table[i][j], ha="center", va="center", color="w")

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')

        problems = self.cfg['problems']['problems']
        plt.xticks(list(range(len(problems))), problems)
        plt.yticks(list(range(len(problems))), problems)

        plt.close(fig)
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        image = image.reshape((3, 1000, 1000))

        self.writer.add_image('Info/Confusion_matrix', image, 0)

    def close(self):
        self.writer.close()
