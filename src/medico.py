from scipy.misc import imread, imsave, imresize
from keras.models import load_model

import tensorflow as tf
import numpy as np

import time
import os

class_names = os.listdir('/home/steven/Data/medico')

model_path = '/home/steven/github/keras-training-system/experiments/densenet169_imagenet_transfer_medico/22/split_0_densenet169_imagenet_transfer_medico_model.h5'

def run():

    test_data_path = '/home/steven/Data/medico_test'
    test_results_path = '../medico_submissions'
    test_results_file_name = f'submission_{len(os.listdir(test_results_path))}'

    file_path = os.path.join(test_results_path, test_results_file_name)
    model = load_model(model_path)

    index_label = { index: class_name for index, class_name in enumerate(class_names) }

    for index, file_name in enumerate(os.listdir(test_data_path)):

        print(f'{index + 1}/{len(os.listdir(test_data_path)) + 1}', end='\r')

        prediction = None

        image = imread(os.path.join(test_data_path, file_name), mode='RGB')
        image = imresize(image, (299, 299, 3))
        image = image.reshape(1, 299, 299, 3)
        image = np.true_divide(image, 255.)

        with tf.get_default_graph().as_default():
            prediction = model.predict(image)[0]

        prediction_index = np.argmax(prediction)
        prediction_label = index_label[prediction_index]
        
        with open( file_path, 'a' if os.path.exists(file_path) else 'w' ) as f:
            f.write(f'{ file_name },{ prediction_label },{ prediction[prediction_index] }\n')

if __name__ == '__main__':

    run()