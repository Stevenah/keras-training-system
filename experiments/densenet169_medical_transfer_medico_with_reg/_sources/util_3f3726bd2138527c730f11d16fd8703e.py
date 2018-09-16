#!/usr/bin/env python

from keras.callbacks import Callback
from keras.models import model_from_json
from scipy.misc import imsave, imread, imresize

import numpy as np
import tensorflow as tf
import keras.backend as K

import os
import cv2
import shutil

def merge_dict_of_lists(dict1, dict2):
  keys = set(dict1).union(dict2)
  no = []
  return dict((k, dict1.get(k, no) + dict2.get(k, no)) for k in keys)

def prepare_dataset(split_dirs, split_index, split_total):
    training_dir = os.path.join('../tmp', 'training_data')
    validation_dir = os.path.join('../tmp', 'validation_data')

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

def get_sub_dirs(path):
    root, *_ = os.walk(path)
    return root[1]

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

    if os.path.exists('../tmp'):
        shutil.rmtree('../tmp')

    split_dirs = []

    for index in range(folds):
        split_dirs.append(os.path.join('../tmp', 'splits', f'split_{index}'))

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