import os, sys, logging, inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from collections import OrderedDict, defaultdict
import random

import torch
import mautil as mu

import util
from util import parser
from models import *
from dataset import gen_ds
#from preprocess import preprocess_data
from copy import deepcopy
gl = globals()


logger = logging.getLogger(__name__)


def kf(args, cls_dict=None):
    if cls_dict is None:
        cls_dict = gl

    train = util.load_data(args)
    args.std_v = np.std(train, 0, keepdims=True).tolist()
    args.mean_v = np.mean(train, 0, keepdims=True).tolist()


    kf = KFold(n_splits=args.kn, shuffle=True, random_state=9527)
    splits = kf.split(train)
    rstate = np.random.RandomState(args.seed)
    kf_result = []
    val_preds, test_preds = [], []

    for name, cls in mu.parse_model_name(args.model_names).items():
        for i in range(args.kn):
            args.seed = rstate.randint(1000000)
            train_inds, val_inds = next(splits)
            if args.kfid is not None:
                if str(i) not in args.kfid.split(' '):
                    continue
            val_data = train[val_inds]
            train_data = train[train_inds]

            logging.info("*************kfid:%s", i)
            model = mu.create_model(name+'_KF'+str(i), args, cls_dict[cls])
            if args.distill_model is not None:
                cfg = model.cfg.copy()
                cfg.model_names = cfg.distill_model
                cfg.restore = True
                distill_model = mu.create_model(cfg.distill_model+ '_KF' + str(i), cfg, cls_dict[cls])
                distill_model.cfg.load(distill_model.gen_fname('cfg.json'))
                distill_model.create_model()
                model.distill_model = distill_model._model.eval()
                model.cfg.distill_d_emodel = distill_model.cfg.d_emodel
                model.cfg.distill_d_dmodel = distill_model.cfg.d_dmodel
            train_ds = gen_ds(model.cfg, 'train', train_data)
            val_ds = gen_ds(model.cfg, 'val', val_data)
            if not args.no_train:
                model.fit(train_ds, val_ds)
            val_pred = None
            if (not args.no_predicting):
                logger.info('start predict_rst')
                model.restore()
                val_pred = model.predict_rst(val_ds, data_type='val')
                val_preds.append(val_pred)
            elif args.no_train:
                val_pred = model.load_predict('_val')
                val_preds.append(val_pred)
            #if args.scoring:
            model.score(val_ds, preds=val_pred)
            #if args.save:
            #    model.save_predict(val_pred, '_val')
            #model.score(val_ds)

        #val_pred = {k: np.concatenate([pred[k] for pred in val_preds]) for k in val_preds[0].keys()}

        #s = util.score(val_pred, len(util.label2id))
        #logging.info('  ***** kf validate score for model:%s, %s', cls, s)


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.batch_size is None:
        args.batch_size = 32

    if not args.debug:
        mu.set_logger(logging.INFO)
    else:
        mu.set_logger(logging.DEBUG)
        args.epochs = 3
        args.n_e_layer = 2
        args.n_d_layer = 2
        args.n_head = 2
        args.d_emodel = 24
        args.d_dmodel = 24
        args.enc_dim = 48
        args.enc_time = 24
        args.verbose = 1
        args.lr = 1e-3
        args.batch_size = 4
        args.num = 20
        args.kn = 2
        args.n_lr_warmup_step = 5
        args.n_lr_decay_step = 10
        args.n_epoch_step = 20
        args.lr_decay_rate = 0.1
        args.ema = 0.9
        #args.no_validate = True
        #args.restore = True
        args.save = True
        #args.save_opt = True
        args.n_keep_ckpt = 2
        #args.n_save_epoch = 1
        #args.save_half = True
        args.use_fp16 = True
        args.data_type = 'train'
        args.method_name = 'kf'
        args.lr_scheduler = 'ld'
        args.train_quantize = True
        args.use_round_loss = True
        args.n_q_bit = 2
        args.mixup = 0.5
        args.vq_dim = 12
        args.vq_groups = 3
        args.kfid = '0'
        args.model_names = 'CSIPlus'
        args.use_mse = True
        args.accumulated_batch_size = 2

    if args.method_name in gl:
        if inspect.isfunction(gl[args.method_name]):
            gl[args.method_name](args)
        else:
            args.model_names = args.method_name
            train(args)
    else:
        logging.error('unknown method : %s', args.method_name)

