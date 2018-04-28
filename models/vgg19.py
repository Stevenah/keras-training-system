from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16

def build(config):

    image_width    = config['image_processing']['image_width']
    image_height   = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    number_of_classes = config['dataset']['number_of_classes']

    base_model = VGG16(
        weights=None,
        include_top=False,
        input_shape=(image_width, image_height, image_channels))

    x = base_model.output
    x = Flatten()(x)

    predictions = Dense(
        units=number_of_classes,
        activation='softmax',
        name='predictions')(x)

    return Model(inputs=base_model.input, outputs=predictions)

    
