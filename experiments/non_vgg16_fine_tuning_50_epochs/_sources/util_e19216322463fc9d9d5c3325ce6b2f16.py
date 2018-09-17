from keras.callbacks import Callback
from keras.models import model_from_json
from scipy.misc import imsave, imread, imresize

from utils.constants import TEMP_PATH

import numpy as np
import tensorflow as tf
import keras.backend as K

import os
import cv2
import shutil


class ModelHelper():

    def __init__(self, model, classes, width, height, channels):

        self.model = model
        self.graph = tf.get_default_graph()

        self.labels = classes

        self.image_width = width
        self.image_height = height
        self.image_channels = channels

    def prepare_image(self, image):
        image = imresize(image, (self.image_width, self.image_height, self.image_channels))
        image = image.reshape(1, self.image_width, self.image_height, self.image_channels)
        return image

    def predict_from_path(self, path):
        image = imread(path, mode='RGB')
        image = self.prepare_image(image)
        image = np.true_divide(image, 255.)
        return self.predict_max(image)

    def predict(self, image):
        with self.graph.as_default():
            return self.model.predict(image)[0]

    def predict_max(self, image):
        return np.argmax(self.predict(image))

def merge_dict_of_lists(dict1, dict2):
  keys = set(dict1).union(dict2)
  no = []
  return dict((k, dict1.get(k, no) + dict2.get(k, no)) for k in keys)

def prepare_dataset(split_dirs, split_index, split_total):
    training_dir   = os.path.join(TEMP_PATH, 'training_data')
    validation_dir = os.path.join(TEMP_PATH, 'validation_data')

    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)

    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)

    os.makedirs(validation_dir)
    os.makedirs(training_dir)

    for index in range(split_total):
        if index == split_index:
            copytree(split_dirs[index], validation_dir)
        else:
            copytree(split_dirs[index], training_dir)

    return training_dir, validation_dir

def save_artifact(experiment, artifact, artifact_name):
    with open(artifact_name, 'w') as f:
        f.write(artifact)
    experiment.add_artifact(artifact_name)
    os.remove(artifact_name)

def get_sub_dirs(path):
    root, *_ = os.walk(path)
    return root[1]

class LogPerformance(Callback):

    def __init__(self, experiment, filename):
        self.filename = filename
        self.experiment = experiment

    def on_epoch_end(self, _, logs={}):
        @self.experiment.capture
        def log_performance(_run, logs):
            if os.path.isfile(self.filename):
                _run.add_artifact(self.filename)
            _run.log_scalar("loss", float(logs.get('loss')))
            _run.log_scalar("accuracy", float(logs.get('acc')))
            _run.log_scalar("val_loss", float(logs.get('val_loss')))
            _run.log_scalar("val_accuracy", float(logs.get('val_acc')))
            _run.result = float(logs.get('val_acc'))
        log_performance(logs=logs)

def pad_string(string, size, fill=" ", edge="|"):
    return f'{edge}{string}{((size - len(string)) *  fill)}{edge}\n'

def copytree(sourceRoot, destRoot):
    if not os.path.exists(destRoot):
        return False
    ok = True
    for path, dirs, files in os.walk(sourceRoot):
        relPath = os.path.relpath(path, sourceRoot)
        destPath = os.path.join(destRoot, relPath)
        if not os.path.exists(destPath):
            os.makedirs(destPath)
        for file in files:
            destFile = os.path.join(destPath, file)
            if os.path.isfile(destFile):
                ok = False
                continue
            srcFile = os.path.join(path, file)
            shutil.copy2(srcFile, destFile)

def split_data_on_suffix(test_suffix, data_dir):

    training_dir = os.path.join(data_dir, 'training')
    validation_dir = os.path.join(data_dir, 'validation')

    class_names = get_sub_dirs(training_dir)

    for class_name in class_names:
        train_class_dir = os.path.join(training_dir, class_name)
        valid_class_dir  = os.path.join(validation_dir, class_name)

        if not os.path.exists(valid_class_dir):
            os.makedirs(valid_class_dir)

        for filename in os.listdir(train_class_dir):
            if os.path.splitext(os.path.basename(filename))[0][-(len(test_suffix)):] == test_suffix:
                file_source = os.path.join(train_class_dir, filename)
                file_dest = os.path.join(valid_class_dir, filename)
                shutil.copy(file_source, file_dest)

    return training_dir, validation_dir

def split_data(folds, data_dir):
    class_names = get_sub_dirs(data_dir)

    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

    split_dirs = []

    for index in range(folds):
        split_dirs.append(os.path.join(TEMP_PATH, 'splits', f'split_{index}'))

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        split_size = len(os.listdir(class_dir)) // folds

        split_start = 0
        split_end   = split_size

        for split_index in range(folds):

            split_dir = os.path.join(split_dirs[split_index], class_name)
            os.makedirs(split_dir)

            for filename in os.listdir(class_dir)[split_start:split_end]:
                file_source = os.path.join(class_dir, filename)
                file_dest = os.path.join(split_dir, filename)
                os.symlink(file_source, file_dest)

            split_start += split_size
            split_end += split_size

    return split_dirs

def copysuffix(split_dir, dest_dir, suffix):
    class_names = get_sub_dirs(split_dir)
    for class_name in class_names:
        split_class_dir = os.path.join(split_dir, class_name)
        dest_class_dir  = os.path.join(dest_dir, class_name)

        if not os.path.exists(dest_class_dir):
            os.makedirs(dest_class_dir)
        
        for filename in os.listdir(split_class_dir):
            if os.path.splitext(os.path.basename(filename))[0][-(len(suffix) + 1):] == f'_{suffix}':
                file_source = os.path.join(split_class_dir, filename)
                file_dest = os.path.join(dest_class_dir, filename)
                shutil.copy(file_source, file_dest)