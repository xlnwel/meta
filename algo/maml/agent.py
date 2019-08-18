import tensorflow as tf

from basic_model.model import Model
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
        self.k = args['inner_sample_per_class']
        self.num_classes = args['num_classes_per_task']
        self.dataset = DataGenerator(args['num_tasks_per_batch'], self.num_classes, self.k + args['num_outer_samples_per_class'])
        super().__init__(name, args, 
                         sess_config=sess_config, 
                         save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device)

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


        

    def _prepare_data(self):
        with tf.name_scope('data'):
            image, label = self.dataset.make_data_tensor(self.training)

            inner_image = tf.slice(image, [0, 0, 0], [-1, self.k * self.num_classes, -1])
            outer_image = tf.slice(image, [0, self.k * self.num_classes, 0], [-1, -1, -1])
            inner_label = tf.slice(label, [0, 0, 0], [-1, self.k * self.num_classes, -1])
            outer_label = tf.slice(label, [0, self.k * self.num_classes, 0], [-1, -1, -1])
            
        return inner_image, outer_image, inner_label, outer_label