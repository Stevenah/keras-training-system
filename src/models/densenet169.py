from keras.applications.densenet import DenseNet169
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers

def build(config):

    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    number_of_classes = config['dataset']['number_of_classes']

    weights = config['model'].get('weights', None)
    print("weights:", weights)
    
    if weights == 'imagenet':
        base_model = DenseNet169(
            input_shape=(image_width, image_height, image_channels),
            weights='imagenet',
            include_top=False)
    else:
        base_model = DenseNet169(
            input_shape=(image_width, image_height, image_channels),
            weights=None,
            include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    predictions = Dense(
        activity_regularizer=regularizers.l1_l2(),
        units=number_of_classes,
        activation='softmax',
        name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    if weights != 'imagenet' and weights is not None:
        print(weights, weights is not 'imagenet')
        model.load_weights(weights)

    return model