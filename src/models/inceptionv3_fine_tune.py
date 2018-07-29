from keras.models import Model
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3

def build(config):
    
    image_width    = config['image_processing']['image_width']
    image_height   = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    number_of_classes = config['dataset']['number_of_classes']

    base_model = InceptionV3(
        input_shape=(image_width, image_height, image_channels),
        weights='imagenet',
        include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    predictions = Dense(number_of_classes, activation='softmax')(x)

    return Model(base_model.input, predictions)