import json
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, model_fname, cfg):
        self.model_fname = model_fname
        self.writer = SummaryWriter(log_dir='runs/' + model_fname)
        self.writer.add_text('Info/Config', json.dumps(cfg), 0)

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

    def close(self):
        self.writer.close()
