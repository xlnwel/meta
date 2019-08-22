import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from collections import deque
from time import time
from pathlib import Path
import tensorflow as tf

from model import MAML
from utility.utils import set_global_seed
from utility.debug_tools import pwc
from run.grid_search import GridSearch


def main(args):
    # you may need this code to train multiple instances on a single GPU
    # sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.allow_growth=True
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    # remember to pass sess_config to Model

    model_train = MAML('MAML', args, training=True, log_tensorboard=True, save=True)
    # model_eval = MAML('MAML', args, training=False, log_tensorboard=True, save=False, reuse=True)

    duration = 200
    for i in range(0, 60000, duration):
        start = time()
        times = deque(maxlen=100)
        model_train.train(i, duration, start, times)
        if i != 0 and i % (10 * duration) == 0:
            # model_eval.evaluate(i // (10 * duration), 200)
            model_train.save()

if __name__ == '__main__':
    set_global_seed()
    args_file = 'args.yaml'

    gs = GridSearch(args_file, main)

    gs()
