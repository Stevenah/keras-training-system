from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

def build(config):
    
    image_width    = config['image_processing']['image_width']
    image_height   = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    number_of_classes = config['dataset']['number_of_classes']
    
    base_model = Xception(
        input_shape=(image_width, image_height, image_channels),
        weights='imagenet',
        include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    predictions = Dense(number_of_classes, activation='softmax')(x)

    return Model(base_model.input, predictions)