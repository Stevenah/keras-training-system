from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

def build(config):
    
    image_width    = config['image_processing']['image_width']
    image_height   = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    number_of_classes = config['dataset']['number_of_classes']

    base_model = ResNet50(
        input_shape=(image_width, image_height, image_channels),
        weights='imagenet',
        include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    predictions = Dense(
        units=number_of_classes,
        activation='softmax',
        name='predictions')(x)

    return Model(base_model.input, predictions)