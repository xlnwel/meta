import tensorflow as tf

from basic_model.model import Model
from utility.tf_utils import stats_summary
from data.data_generator import DataGenerator


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
                 device=None):
        self.training = training
        self.dataset = DataGenerator(args['num_tasks_per_batch'], args['num_classes_per_task'], args['num_samples_per_class'])
        
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
            self.image, self.label = self._prepare_data()

        

    def _prepare_data(self):
        with tf.name_scope('data'):
            image, label = self.dataset.make_data_tensor(self.training)
            
        return image, label