from scipy.misc import imread, imsave, imresize

import tensorflow as tf
import numpy as np

import time
import os

class_names = [ ]

model_path = ''

def run():

    test_data_path = ''
    test_results_path = '../medico_submissions'
    test_results_file_name = f'submission_{len(os.listdir(test_results_paths))}'

    file_path = os.path.join(test_results_path, test_results_file_name)

    index_label = { index: class_name for index, class_name in enumerate(class_names) }

    for file_name in os.listdir(test_data_path):

        prediction = None

        image = imread(os.path.join(class_dir, file_name), mode='RGB')
        image = imresize(image, (image_width, image_height, image_channels))
        image = image.reshape(1, image_width, image_height, image_channels)
        image = np.true_divide(image, 255.)

        with tf.get_default_graph().as_default():
            prediction = model.predict(image)[0]

        prediction_index = np.argmax(prediction)
        prediction_label = index_label[prediction_index]
        
        with open( file_path, 'a' if os.path.exists(file_path) else 'w' ) as f:
            f.write(f'{ file_name },{ prediction_label },{ prediction[prediction_index] }\n')

if __name__ == '__main__':

    run()