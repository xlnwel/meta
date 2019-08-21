from time import time
from collections import deque
import numpy as np
import tensorflow as tf

from basic_model.model import Model
from utility.debug_tools import timeit
from utility.tf_utils import stats_summary
from dataset.data_generator import DataGenerator
from algo.maml.network import Network


class MAML(Model):
    """ Interface """
    def __init__(self, 
                 name, 
                 args,
                 training,
                 sess_config=None, 
                 save=True, 
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None,
                 reuse=None):
        self.training = training
        self.k = args['num_inner_sample_per_class']
        self.num_classes = args['num_classes_per_task']
        self.dataset = DataGenerator(args['num_tasks_per_batch'], self.num_classes, self.k + args['num_outer_samples_per_class'])
        
        super().__init__(name, args, 
                         sess_config=sess_config, 
                         save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device,
                         reuse=reuse)

    def train(self, start_itr, duration, start_time, times):
        times = deque(maxlen=100)
        for i in range(start_itr, start_itr + duration):
            t, (_, loss, acc) = timeit(lambda: self.sess.run([self.network.opt_op, 
                                                            self.network.avg_outer_losses[-1], 
                                                            self.network.avg_outer_accuracies[-1]]))
            times.append(t)
            print(f'\rTime-{(time() - start_itr) / 60:.2f}m Iterator-{i}: loss-{loss:.2f}, accuracy-{acc:.2f}\t\
                    Average {np.mean(times):.3F} seconds per pass',
                  end='')
            print()
            if i == start_itr + duration - 1:
                summary = self.sess.run([self.graph_summary])
                self.writer.add_summary(summary, start_itr + duration - 1)

    def evaluate(self, start_itr, duration):
        for i in range(1, duration + 1):
            loss, acc = self.sess.run([self.network.avg_outer_losses[-1], 
                                        self.network.avg_outer_accuracies[-1]])
            print(f'\rIterator-{i}: loss-{loss:.2f}, accuracy-{acc:.2f}', end='')
            if i == duration:
                summary = self.sess.run([self.graph_summary])
                self.writer.add_summary(summary, start_itr + duration)

    """ Implementation """
    def _build_graph(self):
        with tf.device('/CPU: 0'):
            inner_image, outer_image, inner_label, outer_label = self._prepare_data()

        self.args['network']['num_classes'] = self.num_classes
        self.args['network']['k'] = self.k
        self.args['network']['num_tasks_per_batch'] = self.args['num_tasks_per_batch']
        self.network = Network('Network', self.args['network'], self.graph, 
                                (inner_image, outer_image, inner_label, outer_label), 
                                scope_prefix=self.name, 
                                log_tensorboard=self.log_tensorboard, 
                                log_params=self.log_params)

        self._log_train_info()
        
    def _prepare_data(self):
        with tf.name_scope('data'):
            image, label = self.dataset.make_data_tensor(self.training)

            inner_image = tf.slice(image, [0, 0, 0], [-1, self.k * self.num_classes, -1])
            outer_image = tf.slice(image, [0, self.k * self.num_classes, 0], [-1, -1, -1])
            inner_label = tf.slice(label, [0, 0, 0], [-1, self.k * self.num_classes, -1])
            outer_label = tf.slice(label, [0, self.k * self.num_classes, 0], [-1, -1, -1])
            
        return inner_image, outer_image, inner_label, outer_label

    def _log_train_info(self):
        if not self.log_tensorboard:
            return

        name = 'train_info' if self.training else 'val_info'
        with tf.name_scope(name):
            with tf.name_scope('loss'):
                tf.summary.scalar('task_loss_', self.network.avg_task_loss)
                for i, loss in enumerate(self.network.avg_outer_losses):
                    tf.summary.scalar(f'outer_loss_{i}_', loss)

            with tf.name_scope('accuracy'):
                tf.summary.scalar('task_acc_', self.network.avg_task_accuracy)
                for i, loss in enumerate(self.network.avg_outer_accuracies):
                    tf.summary.scalar(f'outer_loss_{i}_', loss)