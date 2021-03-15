import gzip
import pickle

import pyxis as px
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class ChunkedDataLoader:
    # TODO NOT FINISHED

    def __init__(self, cfg):
        self.cfg = cfg
        self.chunks = cfg['problems']['chunks']
        self.batch_size = cfg['model']['batch_size']
        self.train_eval_split = cfg['model']['train_eval_split']
        self.dataset_id = cfg['dataset_id']
        self.current_chunk = 0
        self.last_loaded_chunk = -1

    def _load_chunk(self):
        fname = 'datasets/%s.%d.pickle.gz' % (
            self.dataset_id, self.current_chunk
        )
        with gzip.open(fname, 'rb') as f:
            data, labels = pickle.load(f)

        tensor_x = torch.Tensor(data)
        tensor_y = torch.Tensor(labels)
        dataset = TensorDataset(tensor_x, tensor_y.long())

        train_size = int(self.train_eval_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.cfg['model']['shuffle_data']
        )
        self.test_data_loader = DataLoader(
            test_dataset,
            batch_size=1
        )
        self.last_loaded_chunk = self.current_chunk
        return self.train_data_loader, self.test_data_loader

    def get_current_chunk(self):
        if self.current_chunk != self.last_loaded_chunk:
            return self._load_chunk()
        return self.train_data_loader, self.test_data_loader

    def increment(self):
        self.current_chunk += 1
        if self.current_chunk == self.chunks:
            return False
        return True


# Environment can't be pickled.
# class LMDBDataSet(pxt.TorchDataset):
#     def __init__(self, cfg):
#         self.dataset_id = cfg['dataset_id']
#         super(LMDBDataSet, self).__init__('datasets/' + self.dataset_id)
#
#     def __getitem__(self, idx):
#         sample = self.db[idx]
#         return sample['input'], sample['target']


class LMDBDataSet(torch.utils.data.Dataset):
    def __init__(self, cfg, reverse):
        self.dataset_id = cfg['dataset_id']
        self.dirpath = 'datasets/' + self.dataset_id
        self.reverse = reverse

    def __len__(self):
        if not hasattr(self, "len_"):
            db = px.Reader(self.dirpath, lock=False)
            self.len_ = len(db)
            db.close()
        return self.len_

    def __getitem__(self, key):
        if not hasattr(self, "db"):
            self.db = px.Reader(self.dirpath, lock=False)

        sample = self.db[key]
        if self.reverse:
            if isinstance(sample, list):
                return [(s['input'], s['target'], s['prob']) for s in sample]
            return sample['input'], sample['target'], sample['prob']
        else:
            return sample['input'], sample['target']

    def __repr__(self):
        return str(self.db)


class LMDBDataLoader:
    def __init__(self, cfg, reverse=False, part=None):
        self.cfg = cfg
        self.dataset_id = cfg['dataset_id']
        self.dataset = LMDBDataSet(cfg, reverse)
        if part is not None:
            self.dataset = self.dataset[slice(*part)]
        self.batch_size = cfg['model']['batch_size']
        self.train_eval_split = cfg['model']['train_eval_split']

        train_size = int(self.train_eval_split * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )

        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=cfg['model']['shuffle_data'],
            num_workers=4
        )
        self.test_data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4
        )
