import os, logging
import scipy.io as sio
import numpy as np

import mautil as mu

logger = logging.getLogger(__name__)

parser = mu.TrainArgParser(conflict_handler='resolve')
parser.add_argument("-dim1", type=int, default=24)
parser.add_argument("-dim2", type=int, default=16)
parser.add_argument("-dim3", type=int, default=2)
parser.add_argument("-n_q_bit", type=int, help="num of bits to represent float number")
parser.add_argument("-mixq_factor", type=float)
parser.add_argument("-ds", "--dataset", default='csi', help="dataset name")
parser.add_argument("-backbone")
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-pt_model")
parser.add_argument("-task_cnt", type=int, default=4)
parser.add_argument("-data_type", default='train')
parser.add_argument("-case", default='uncased')
parser.add_argument("-min_cnt", default=0, type=int, help="min count for vocabulary")
parser.add_argument("-use_avg", action="store_true")
parser.add_argument("-train_quantize", action="store_true")
parser.add_argument("-use_round_loss", action="store_true")
parser.add_argument("-n_head", type=int)
parser.add_argument("-round_weight", type=float, default=1.0)
parser.add_argument("-min_temp", type=float, default=0.1)
parser.add_argument("-max_temp", type=float, default=2.0)
parser.add_argument("-min_loss", type=float, default=0.09)
parser.add_argument("-vq_groups", type=int)
parser.add_argument("-vq_depth", type=int)
parser.add_argument("-vq_proj_factor", type=int)
parser.add_argument("-vq_dim", type=int)
parser.add_argument("-temp_decay_rate", type=float, default=0.9999)
parser.add_argument("-mixup", type=float, default=0)
parser.add_argument("-share_weights", action="store_true", default=None)
parser.add_argument("-restore_opt", action="store_true")
parser.add_argument("-is_test", action="store_true")
parser.add_argument("-batch_first", action="store_true")
parser.add_argument("-restore_pretrain", action="store_true")
parser.add_argument("-use_mse", action="store_true")
parser.add_argument("-no_init", action="store_true")
parser.add_argument("-shift_dim", action="store_true", default=None)
parser.add_argument("-do_dyq", action="store_true", default=None)
parser.add_argument("-no_deq", action="store_true", default=None)
parser.add_argument("-use_bn", action="store_true", default=None)
parser.add_argument("-use_in", action="store_true", default=None)
parser.add_argument("-use_mm", action="store_true", default=None)
parser.add_argument("-use_power", action="store_true", default=None)
parser.add_argument("-minus_mm", action="store_true")
parser.add_argument("-ffd")
parser.add_argument("-n_repeat", type=int, default=1)
parser.add_argument("-enc_dim", type=int)
parser.add_argument("-enc_time", type=int)
parser.add_argument("-d_emodel", type=int)
parser.add_argument("-d_dmodel", type=int)
parser.add_argument("-momentum", type=float)
parser.add_argument("-d_eff", type=int)
parser.add_argument("-d_dff", type=int)
parser.add_argument("-n_ehead", type=int)
parser.add_argument("-n_dhead", type=int)
parser.add_argument("-dp", type=float)
parser.add_argument("-n_e_layer", type=int)
parser.add_argument("-n_d_layer", type=int)
parser.add_argument("-distill_model", default=None)
parser.add_argument("-distill_factor", type=int, default=1)
parser.add_argument("-initializer_range", type=float)
parser.add_argument("-activation")


def load_data(args):
    if args.debug:
        data = mu.load_dump(os.path.join(args.data_dir, args.dataset, 'debug.dump'))
    else:
        mat = sio.loadmat(os.path.join(args.data_dir, args.dataset, 'H_4T4R.mat'))
        data = mat['H_4T4R']
        data = data.astype('float32')
        data = np.reshape(data, (-1, 24, 16, 2))
    data = data[:args.num]
    #data = np.transpose(data, (0, 3, 1, 2))
    logger.info('num is %s', len(data))
    return data


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def score(csi, dec):
    s = 1-NMSE(csi, dec)
    return s


if __name__ == "__main__":
    args = parser.parse_args([])

