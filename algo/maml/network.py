import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from utility.utils import pwc
from utility import tf_utils
from basic_model.model import Module

class Network(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph, 
                 input_label,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.input_label = input_label

        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        num_tasks_per_batch = self.args['num_tasks_per_batch']
        num_inner_updates = self.args['num_inner_updates']
        self.weights = _conv_weights(self.args['num_channels'], self.args['num_classes'])

        dtype = [tf.float32, tf.float32, [tf.float32] * num_inner_updates, [tf.float32] * num_inner_updates]
        # to construct the network for later reuse
        self._task_metalearn([x[0] for x in self.input_label], reuse=False)
        # here we implicitly reuse all parameters constructed in the previous step
        results = tf.map_fn(self._task_metalearn, elems=self.input_label, dtype=dtype, 
                            parallel_iterations=num_tasks_per_batch, name='map_fn')
        task_losses, task_accuracies, outer_losses, outer_accuracies = results

        with tf.variable_scope('train'):
            self.avg_task_loss = tf.reduce_sum(task_losses) / num_tasks_per_batch
            self.avg_outer_losses = [tf.reduce_sum(loss) / num_tasks_per_batch for loss in outer_losses]
            self.avg_task_accuracy = tf.reduce_sum(task_accuracies) / num_tasks_per_batch
            self.avg_outer_accuracies = [tf.reduce_sum(acc) / num_tasks_per_batch for acc in outer_accuracies]

            self.opt_op = self._optimization_op(self.avg_outer_losses[-1])
            # optimizer = tf.train.AdamOptimizer(float(self.args['meta_learning_rate']))
            # gvs = optimizer.compute_gradients(self.avg_outer_losses[-1])
            # gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
            # self.opt_op = optimizer.apply_gradients(gvs)

    def _task_metalearn(self, inputs, reuse=True):
        """construct graph for a task with data inputs
        
        Arguments:
            inputs {tuple} -- task-specific data: inner_image, outer_image, inner_label, outer_label
        
        Keyword Arguments:
            reuse {bool} -- whether to reuse network (default: {True})
        """
        def loss_fn(logits, label, name):
            with tf.name_scope(name):
                # TODO: use losses.softmax...
                return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label) / self.args['k']
        
        second_derivatives = self.args['second_derivatives']
        inner_lr = float(self.args['inner_learning_rate'])
        num_inner_updates = self.args['num_inner_updates']

        inner_image, outer_image, inner_label, outer_label = inputs
        outer_ouputs, outer_losses, outer_accuracies = [], [], []

        weights = self.weights
        # we don't distinguish the original weights and updated weights for simplicity
        fast_weights = weights

        with tf.variable_scope('meta_net', reuse=reuse):
            for i in range(num_inner_updates):
                initial_iteration = (i == 0)
                inner_output = self._forward_conv(inner_image, fast_weights, reuse=reuse or not initial_iteration)
                inner_loss = loss_fn(inner_output, inner_label, f'inner_loss_{i}')

                if initial_iteration:
                    task_output, task_loss = inner_output, inner_loss

                grads = tf.gradients(inner_loss, list(fast_weights.values()))
                if not second_derivatives:
                    # prohibit second derivative through grads
                    grads = [tf.stop_gradient(grad) for grad in grads]
                fast_weights = dict(zip(fast_weights.keys(), [val - inner_lr * grad for val, grad in zip(fast_weights.values(), grads)]))
                outer_ouput = self._forward_conv(outer_image, fast_weights, reuse=True)
                outer_ouputs.append(outer_ouput)
                outer_losses.append(loss_fn(outer_ouput, outer_label, f'outer_loss_{i}'))

            task_accuracy = tc.metrics.accuracy(tf.argmax(task_output, 1), tf.argmax(inner_label, 1))
            outer_accuracies = [tc.metrics.accuracy(tf.argmax(out, 1), tf.argmax(outer_label, 1)) for out in outer_ouputs]

        return [task_loss, task_accuracy, outer_losses, outer_accuracies]

    def _forward_conv(self, x, weights, reuse=None):
        norm = self.args['norm']
        max_pool = self.args['max_pool']

        with tf.variable_scope('net_params', reuse=reuse):
            x = _conv_block(x, weights['conv1'], weights['b1'], norm, max_pool, 1)
            x = _conv_block(x, weights['conv2'], weights['b2'], norm, max_pool, 2)
            x = _conv_block(x, weights['conv3'], weights['b3'], norm, max_pool, 3)
            x = _conv_block(x, weights['conv4'], weights['b4'], norm, max_pool, 4)

            x = tf.reshape(x, [-1, np.prod(x.shape.as_list()[1:])])

            x = x @ weights['w5'] + weights['b5']

        return x

def _conv_block(x, kernel, bias, norm, max_pool, block_idx):
    if norm == 'batch_norm':
        norm = lambda x: tf.layers.batch_normalization(x, training=True)
    elif norm =='layer_norm':
        norm = tc.layers.layer_norm
    elif norm is None:
        pwc('No normalization is used')
    else:
        raise NotImplementedError

    strides = [1, 1, 1, 1] if max_pool else [1, 2, 2, 1]
    
    with tf.variable_scope(f'conv_block_{block_idx}'):
        x = tf.nn.conv2d(x, kernel, strides, 'SAME') + bias
        x = norm(x)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    return x

def _conv_weights(num_channels, num_classes):
    weights = {}

    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    fc_initializer = tf.contrib.layers.xavier_initializer()
    k = 3

    with tf.variable_scope('weights', reuse= tf.AUTO_REUSE):
        weights['conv1'] = tf.get_variable('conv1w', [k, k, 3, num_channels],  initializer=conv_initializer)
        weights['b1'] = tf.get_variable('conv1b', initializer=tf.zeros([32]))
        weights['conv2'] = tf.get_variable('conv2w', [k, k, 32, num_channels], initializer=conv_initializer)
        weights['b2'] = tf.get_variable('conv2b', initializer=tf.zeros([32]))
        weights['conv3'] = tf.get_variable('conv3w', [k, k, 32, num_channels], initializer=conv_initializer)
        weights['b3'] = tf.get_variable('conv3b', initializer=tf.zeros([32]))
        weights['conv4'] = tf.get_variable('conv4w', [k, k, 32, num_channels], initializer=conv_initializer)
        weights['b4'] = tf.get_variable('conv4b', initializer=tf.zeros([32]))

        # assumes max pooling
        weights['w5'] = tf.get_variable('fc1w', [32 * 5 * 5, num_classes], initializer=fc_initializer)
        weights['b5'] = tf.get_variable('fc1b', initializer=tf.zeros([num_classes]))

        return weights
