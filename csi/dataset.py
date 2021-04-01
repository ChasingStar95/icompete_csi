import os, sys, logging
from tqdm import tqdm
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from glob import glob
from functools import partial
from itertools import islice

import util
import mautil as mu

import torch


logger = logging.getLogger(__name__)





class DatasetMix():
    def __init__(self, cfg, data_type, data):
        self.cfg = cfg
        self.data_type = data_type
        self.data = data
        self.inds = np.arange(len(data))

    def __iter__(self):
        if self.data_type=='train':
            np.random.shuffle(self.inds)
        if self.data_type =='train':
            inds = np.concatenate([self.inds]*self.cfg.n_repeat)
        else:
            inds = self.inds
        for batch_id in np.arange(0, len(inds), self.cfg.batch_size):
            batch_inds = inds[batch_id:batch_id+self.cfg.batch_size]
            if len(batch_inds)!=self.cfg.batch_size and self.data_type=='train':
                break
            else:
                yield self.gen_batch(batch_inds)

    def __getitem__(self, index):
        item = self.getitem(index)
        item = self.item2tensor(item)
        return item

    def getitem(self, index):
        item = {'csi': self.data[index]}
        return item

    def gen_batch(self, inds):
        data = self.data[inds]
        batch = {'csi': torch.tensor(data)}
        return batch

    def item2tensor(self, item):
        for k, v in item.items():
            item[k] = torch.tensor(v)
        return item


class Dataset(DatasetMix, torch.utils.data.Dataset):
    def __len__(self):
        return len(self.data)


def collate_fn(batch, mixup=0, data_type='train'):
    batch = {k: torch.stack([dic[k] for dic in batch]) for k in batch[0]}
    num = int(batch['csi'].shape[0]*mixup)
    if num>1:
        x1 = batch['csi'][:num]
        x2 = torch.cat([x1[1:], x1[:1]], 0)
        w = torch.tensor(np.random.beta(1, 1, (x1.shape[0], 1, 1, 1)).astype(np.float32))
        batch['csi'] = torch.cat([w*x1 + (1-w)*x2, batch['csi'][num:]], 0)
    return batch


def gen_ds(args, data_type, data):
    drop_last, shuffle, num_workers, sampler, batch_size, collate_func = False, False, args.n_dl_worker, None, args.batch_size, None
    if data_type=='train':
        drop_last, shuffle = True, True
        collate_func = partial(collate_fn, mixup=args.mixup, data_type=data_type)
    ds = Dataset(args, data_type, data)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_func)
    return dl


if __name__ == '__main__':
    import mautil as mu
    from util import *
    args = parser.parse_args([])
    args.dataname = 'csi'
    args.batch_size=512
    data = util.load_data(args)
    dl = gen_ds(args, 'train', data)
    for batch in tqdm(dl):
        pass



