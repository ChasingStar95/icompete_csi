import os, logging, pandas as pd, re
import numpy as np
from util import parser
import util


import mautil as mu


logger = logging.getLogger(__name__)




def preprocess_data(args):
    train = util.load_data(args)
    min_v = np.min(train, 0, keepdims=True).tolist()
    max_v = np.max(train, 0, keepdims=True).tolist()
    std_v = np.std(train, 0, keepdims=True).tolist()
    mean_v = np.mean(train, 0, keepdims=True).tolist()
    mu.dump((min_v, max_v, std_v, mean_v), os.path.join(args.output_dir, 'stats.dump'))
    logger.info('stats done')
#    logger.info('min_v:%s', min_v)
#    logger.info('max_v:%s', max_v)
#    logger.info('std_v:%s', std_v)
#    logger.info('mean_v:%s', mean_v)



if __name__ == '__main__':
    gl = globals()
    args = parser.parse_args()
    if args.method_name is None:
        args.method_name = 'preprocess_data'

    if not args.debug:
        mu.set_logger(logging.INFO)
    else:
        mu.set_logger(logging.DEBUG)
        args.method_name = 'preprocess_data'
        args.data_type = 'train'
        args.task_cnt = 1
        args.n_fold = 2
        args.num = 10000
        args.save = True

    if args.data_type not in ['train', 'val', 'test']:
        raise Exception('unknown data type')

    if args.method_name in gl:
        gl[args.method_name](args)
    else:
        logging.error('unknown method name: %s', args.method_or_model)
