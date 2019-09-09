import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import pickle
import numpy as np
import tensorflow as tf
import tqdm

from utility.image_processing import image_dataset, save_image
from utility.utils import pwc


def get_images_labels(paths, labels, nb_samples=None, shuffle=False):
	if nb_samples is not None:
		sampler = lambda x: random.sample(x, nb_samples)
	else:
		sampler = lambda x: x
	images_labels = [(os.path.join(path, image), lb) \
	                for path, lb in zip(paths, labels) \
	                for image in sampler(os.listdir(path))]
	if shuffle:
		random.shuffle(images_labels)
	return images_labels


class DataGenerator:
    def __init__(self, num_tasks_per_batch, num_classes_per_task, num_samples_per_class, 
                metatrain_folder, metaval_folder, total_batch_num=200000):
        # number of tasks per batchå
        self.num_tasks_per_batch = num_tasks_per_batch      # T
        # number of image samples per class
        self.num_classes_per_task = num_classes_per_task    # C
        self.num_samples_per_class = num_samples_per_class  # S
        self.batch_size = num_tasks_per_batch * num_classes_per_task * num_samples_per_class
        self.image_size = (84, 84)
        self.total_batch_num = total_batch_num

        self.metatrain_folders = [os.path.join(metatrain_folder, label) \
                                for label in os.listdir(metatrain_folder) \
                                if os.path.isdir(os.path.join(metatrain_folder, label)) \
		                        ]
        self.metaval_folders = [os.path.join(metaval_folder, label) \
                                for label in os.listdir(metaval_folder) \
                                if os.path.isdir(os.path.join(metaval_folder, label)) \
                                ]
        self.rotations = [0]

    def make_data_tensor(self, training=True):
        if training:
            folders = self.metatrain_folders
            num_total_batches = self.total_batch_num
        else:
            folders = self.metaval_folders
            num_total_batches = 600

        if training and os.path.exists('filelist.pkl'):
            labels = np.arange(self.num_classes_per_task).repeat(self.num_samples_per_class).tolist()
            with open('filelist.pkl', 'rb') as f:
                all_filenames = pickle.load(f)
                pwc('Load file names from filelist.pkl', 'magenta')
        else:
            all_filenames = []
            for _ in tqdm.tqdm(range(num_total_batches), 'generating episodes'):
                # sample data for each task
                sampled_folders = random.sample(folders, self.num_classes_per_task)
                random.shuffle(sampled_folders)

                # for each task we have num_classes_per_task categories
                # the same image may have different labels when it appears in different tasks
                images_labels = get_images_labels(sampled_folders, 
                                                  range(self.num_classes_per_task), 
                                                  nb_samples=self.num_samples_per_class, 
                                                  shuffle=False)

                filenames, labels = list(zip(*images_labels))
                all_filenames += filenames
            
            if training:
                with open('filelist.pkl', 'wb') as f:
                    pickle.dump(all_filenames, f)
                    pwc('Save all file names to filelist.pkl', 'magenta')

        _, images = image_data`set(all_filenames, self.batch_size, self.image_size, norm_range=[0, 1], shuffle=False)
        num_samples_per_task = self.num_classes_per_task * self.num_samples_per_class
        all_image_batches, all_label_batches = [], []

        # shuffle image&labels for each task
        for i in range(self.num_tasks_per_batch):
            # images for the current task
            task_images = images[i * num_samples_per_task: (i + 1) * num_samples_per_task]

            # as all labels of all task are the same, which is 0, 0, ..., 1, 1, ..., 2, 2, ..., 3, 3, ..., 4, 4, ...
            label_batch = tf.convert_to_tensor(labels)
            task_image_list, task_label_list = [], []
            
            # shuffle [k, k+S, k+2*S, k+3*S, k+4*S]
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes_per_task)
                class_idxs = tf.random_shuffle(class_idxs)
                class_idxs = class_idxs * self.num_samples_per_class + k

                task_image_list.append(tf.gather(task_images, class_idxs))
                task_label_list.append(tf.gather(label_batch, class_idxs))

            task_image_list = tf.concat(task_image_list, 0) # [C*S]
            task_label_list = tf.concat(task_label_list, 0) # [C*S, 84, 84, 3]
            all_image_batches.append(task_image_list)
            all_label_batches.append(task_label_list)

        # [4, C*S, 84*84*3]
        all_image_batches = tf.stack(all_image_batches)
        # [4, C*S]
        all_label_batches = tf.stack(all_label_batches)
        # [4, C*S, C]
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes_per_task)

        pwc('Data pipeline has been constructed!', 'magenta')
        return all_image_batches, all_label_batches


if __name__ == '__main__':
    dg = DataGenerator(5, 2, 4, 500, 'dataset/miniimagenet/train', 'dataset/miniimagenet/test')
    image, label = dg.make_data_tensor()

    with tf.Session() as sess:
        img, lb = sess.run([image, label])
        pwc(f'{img.shape}')
        lb = np.argmax(lb, 2)
        print(lb)
        pwc(f'{lb.shape}')
        # comment out tf.one_hot above to enable the following test 
        for i in range(5):
            save_image(img[i][lb[i]==1], f'./data_generator_test/img{i}.png')
