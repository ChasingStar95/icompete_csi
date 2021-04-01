import os, sys, logging
sys.path.insert(-1, '.')
from collections import OrderedDict
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from mautil.pt import PTModel
import mautil as mu

import nn
import util

logger = logging.getLogger(__name__)


class Model(PTModel):
    cfg = PTModel.cfg.copy()
    cfg.batch_keys = ['csi', 'dec']
    cfg.verbose = 256
    cfg.to_list = False
    cfg.do_concat = True
    cfg.is_test = False
    cfg.n_keep_ckpt = 10
    cfg.share_weights = False
    cfg.do_dyq = False
    cfg.use_bn = False
    cfg.use_in = False
    cfg.use_mm = False
    cfg.no_deq = False
    cfg.use_power = False
    cfg.momentum = 0.1

    cfg.opt = 'torch.optim.AdamW'
    cfg.opt_paras = {'weight_decay':1e-6}
    cfg.epochs = 100
    cfg.lr_scheduler = 'ld'
    cfg.lr = 1e-4
    cfg.ffd = ''
    cfg.n_lr_warmup_step = 1000
    cfg.pt_model = 'CSI'
    cfg.encoder = 'Encoder'
    cfg.decoder = 'Decoder'

    cfg.d_emodel = None
    cfg.d_dmodel = None
    cfg.d_eff = 8
    cfg.d_dff = 8
    cfg.n_ehead = 2
    cfg.n_dhead = 2
    cfg.dropout = 0.0
    cfg.n_e_layer = 2
    cfg.n_d_layer = 2
    cfg.activation = 'relu'
    cfg.initializer_range = 0.02
    cfg.shift_dim = False
    cfg.distill_model = None

    cfg.n_q_bit = 4
    cfg.mixq_factor = 0
    cfg.enc_dim = 128

    def __init__(self, name, cfg={}):
        super(Model, self).__init__(name, cfg)
        self.cfg.n_bit = self.cfg.enc_dim*self.cfg.n_q_bit
        if self.cfg.d_emodel is None:
            self.cfg.d_emodel = self.cfg.enc_dim
        if self.cfg.d_dmodel is None:
            self.cfg.d_dmodel = self.cfg.enc_dim


    def create_core_model(self, **kwargs):
        return getattr(nn, self.cfg.pt_model)(self.cfg)

    def fit_batch(self, batch, step=None, phase='train', model=None, opt=None, lr_scheduler=None):
        if self.cfg.distill_model is not None and phase == 'train':
            batch['return_hidden'] = True
            outputs, _ = super(Model, self).fit_batch(batch, step=step, phase='test', model=self.distill_model)
            batch['enc_hidden'] = self.distill_model.encoder.encoder.enc_hidden
            batch['enc_att'] = self.distill_model.encoder.encoder.enc_att
            batch['enc_enc'] = self.distill_model.encoder.enc_enc
            batch['dec_hidden'] = self.distill_model.decoder.decoder.enc_hidden
            batch['dec_att'] = self.distill_model.decoder.decoder.enc_att
        return super().fit_batch(batch, step, phase, model, opt, lr_scheduler)


    def restore(self, restore_epoch=None, epoch=None, model_dir=None):
        if self._model is None:
            self.create_model()

        if not self._restored:
            #encoder_save_path = self.gen_fname('encoder.pth.tar')
            #decoder_save_path = self.gen_fname('decoder.pth.tar')
            if self.cfg.do_dyq:
                self.do_dyq()

            #self._restored_epoch, (encoder_save_path, decoder_save_path) = self.cfg.saved_ckpts[-1]
            (encoder_save_path, decoder_save_path), self._restored_epoch = self.get_checkpoint_path(model_dir, restore_epoch)
            encoder_ckpt = torch.load(encoder_save_path, map_location=torch.device('cpu'))
            decoder_ckpt = torch.load(decoder_save_path, map_location=torch.device('cpu'))
            if 'optimizer_state_dict' in encoder_ckpt and self.cfg.restore_opt:
                self._opt.load_state_dict(encoder_ckpt['optimizer_state_dict'])
                self.info('opt restored from %s', encoder_save_path)
            if 'lr_scheduler_state_dict' in encoder_ckpt and self.cfg.restore_opt:
                self._lr_scheduler.load_state_dict(encoder_ckpt['lr_scheduler_state_dict'])
                self.info('lr scheduler restored from %s', encoder_save_path)
            if 'ema_state_dict' in encoder_ckpt:
                self.load_state_dict(self._ema_model.ema, encoder_ckpt['ema_state_dict'])
                self.info('ema model restored from %s', encoder_save_path)
            self.load_state_dict(self._model.encoder, encoder_ckpt['state_dict'])
            self.load_state_dict(self._model.decoder, decoder_ckpt['state_dict'])
            if self._ema_model is not None:
                self.load_state_dict(self._ema_model.ema.encoder, encoder_ckpt['state_dict'])
                self.load_state_dict(self._ema_model.ema.decoder, decoder_ckpt['state_dict'])

            if 'rng_state' in encoder_ckpt:
                torch.set_rng_state(encoder_ckpt['rng_state'])
            self._restored = True
            self.info('model restored from %s, %s', encoder_save_path, decoder_save_path)


    def do_dyq(self):
        self._model.encoder = torch.quantization.quantize_dynamic(self._model.encoder.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8)
        self._model.decoder = torch.quantization.quantize_dynamic(self._model.decoder.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8)
        if self._ema_model is not None:
            self._ema_model.ema.encoder = torch.quantization.quantize_dynamic(self._ema_model.ema.encoder.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8)
            self._ema_model.ema.decoder = torch.quantization.quantize_dynamic(self._ema_model.ema.decoder.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8)


    def save(self, global_step=None, save_path=None, epoch=None, save_opt=False, **kwargs):
        self.cfg.saved_step = global_step
        encoder_save_path = self.gen_fname('encoder.pth.tar-{}'.format(epoch))
        decoder_save_path = self.gen_fname('decoder.pth.tar-{}'.format(epoch))
        to_save_model = self._model
        if not self.cfg.save_ema and not self.cfg.save_opt and self._ema_model is not None:
            to_save_model = self._ema_model.ema # switch ema
        if self.cfg.save_half:
            to_save_model = mu.pt.models.deepcopy(to_save_model).half()
        if self.cfg.do_dyq:
            to_save_model = torch.quantization.quantize_dynamic(to_save_model.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8)
        encoder_state_dict = {
                      'state_dict': to_save_model.encoder.state_dict(),
                      'rng_state':torch.get_rng_state(),
                      }
        decoder_state_dict = {
                              'state_dict': to_save_model.decoder.state_dict(),
                              'rng_state': torch.get_rng_state(),
        }
        if save_opt:
            #assert self.cfg.save_ema or self._ema_model is None, "should save ema for resume training"
            encoder_state_dict['optimizer_state_dict'] = self._opt.state_dict()
            if self._lr_scheduler is not None:
                encoder_state_dict['lr_scheduler_state_dict'] = self._lr_scheduler.state_dict()
        if (self.cfg.save_ema or self.cfg.save_opt) and self._ema_model is not None:
            encoder_state_dict['ema_state_dict'] = self._ema_model.ema.state_dict()


        self.update_saved_ckpt([encoder_save_path, decoder_save_path], epoch)
        if self.cfg.use_tpu:
            if mu.pt.models.xm.is_master_ordinal():
                super(PTModel, self).save()
        else:
            super(PTModel, self).save()

        if self.cfg.use_tpu:
            mu.pt.models.xm.save(encoder_state_dict, encoder_save_path)
            mu.pt.models.xm.save(decoder_state_dict, decoder_save_path)
        else:
            torch.save(encoder_state_dict, encoder_save_path)
            torch.save(decoder_state_dict, decoder_save_path)
        self.info("Model saved to file encoder:{}, decoder:{}".format(encoder_save_path, decoder_save_path))

    def predict_rst(self, ds, data_type='val', preds=None):
        if preds is None:
            preds = self.predict(ds)
        return preds

    def score(self, ds, data_type='val', preds=None):
        if preds is None:
            preds = self.predict_rst(ds, data_type, preds)
        s = util.score(preds['csi'], preds['dec'])
        logger.info('score is %s', s)
        return s

    def _should_stop(self, best_val_loss, val_loss, best_epoch=-1, current_epoch=-1):
        if super()._should_stop(best_val_loss, val_loss, best_epoch, current_epoch) or (val_loss<self.cfg.min_loss and not self.cfg.debug):
            return True
        else:
            return False


class CSI(Model):
    cfg = Model.cfg.copy()

class CSIPlus(CSI):
    cfg = CSI.cfg.copy()
    cfg.encoder = 'EncoderPlus'
    cfg.decoder = 'DecoderPlus'
