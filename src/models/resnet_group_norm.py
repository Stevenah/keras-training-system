'''
Squeeze-and-Excitation ResNets

References:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    - []() # added when paper is published on Arxiv
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import add
from keras.layers import multiply
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
import importlib

from layers.GroupNormalization import GroupNormalization

WEIGHTS_PATH = ""
WEIGHTS_PATH_NO_TOP = ""

def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape

def ResNet(input_shape=None,
           initial_conv_filters=64,
           depth=[3, 4, 6, 3],
           filters=[64, 128, 256, 512],
           width=1,
           bottleneck=False,
           weight_decay=1e-4,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000):
    """ Instantiate the ResNet architecture. Note that ,
        when using TensorFlow for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The model are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            initial_conv_filters: number of features for the initial convolution
            depth: number or layers in the each block, defined as a list.
                ResNet-50  = [3, 4, 6, 3]
                ResNet-101 = [3, 6, 23, 3]
                ResNet-152 = [3, 8, 36, 3]
            filter: number of filters per block, defined as a list.
                filters = [64, 128, 256, 512
            width: width multiplier for the network (for Wide ResNets)
            bottleneck: adds a bottleneck conv to reduce computation
            weight_decay: weight decay (l2 norm)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or `imagenet` (trained
                on ImageNet)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `tf` dim ordering)
                or `(3, 224, 224)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    assert len(depth) == len(filters), "The length of filter increment list must match the length " \
                                       "of the depth list."

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _create_resnet(classes, img_input, include_top, initial_conv_filters,
                       filters, depth, width, bottleneck, weight_decay, pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnext')

    # load weights

    return model


def ResNet18(input_shape=None,
             width=1,
             bottleneck=False,
             weight_decay=1e-4,
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling=None,
             classes=1000):
    return ResNet(input_shape,
                  depth=[2, 2, 2, 2],
                  width=width,
                  bottleneck=bottleneck,
                  weight_decay=weight_decay,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes)


def ResNet34(input_shape=None,
             width=1,
             bottleneck=False,
             weight_decay=1e-4,
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling=None,
             classes=1000):
    return ResNet(input_shape,
                  depth=[3, 4, 6, 3],
                  width=width,
                  bottleneck=bottleneck,
                  weight_decay=weight_decay,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes)


def ResNet50(input_shape=None,
             width=1,
             bottleneck=True,
             weight_decay=1e-4,
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling=None,
             classes=1000):
    return ResNet(input_shape,
                  width=width,
                  bottleneck=bottleneck,
                  weight_decay=weight_decay,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes)


def ResNet101(input_shape=None,
              width=1,
              bottleneck=True,
              weight_decay=1e-4,
              include_top=True,
              weights=None,
              input_tensor=None,
              pooling=None,
              classes=1000):
    return ResNet(input_shape,
                  depth=[3, 6, 23, 3],
                  width=width,
                  bottleneck=bottleneck,
                  weight_decay=weight_decay,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes)


def ResNet154(input_shape=None,
              width=1,
              bottleneck=True,
              weight_decay=1e-4,
              include_top=True,
              weights=None,
              input_tensor=None,
              pooling=None,
              classes=1000):
    return ResNet(input_shape,
                  depth=[3, 8, 36, 3],
                  width=width,
                  bottleneck=bottleneck,
                  weight_decay=weight_decay,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes)


def _resnet_block(input, filters, k=1, strides=(1, 1)):
    ''' Adds a pre-activation resnet block without bottleneck layers

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = GroupNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or init._keras_shape[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = GroupNormalization(axis=channel_axis, gamma_initializer='zeros')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    m = add([x, init])
    return m


def _resnet_bottleneck_block(input, filters, k=1, strides=(1, 1)):
    ''' Adds a pre-activation resnet block with bottleneck layers

    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    bottleneck_expand = 4

    x = GroupNormalization(axis=channel_axis)(input)
    x = Activation('relu')(x)

    if strides != (1, 1) or init._keras_shape[channel_axis] != bottleneck_expand * filters * k:
        init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = GroupNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = GroupNormalization(axis=channel_axis, gamma_initializer='zeros')(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    m = add([x, init])
    return m


def _create_resnet(classes, img_input, include_top, initial_conv_filters, filters,
                   depth, width, bottleneck, weight_decay, pooling):
    '''Creates a SE ResNet model with specified parameters
    Args:
        initial_conv_filters: number of features for the initial convolution
        include_top: Flag to include the last dense layer
        filters: number of filters per block, defined as a list.
            filters = [64, 128, 256, 512
        depth: number or layers in the each block, defined as a list.
            ResNet-50  = [3, 4, 6, 3]
            ResNet-101 = [3, 6, 23, 3]
            ResNet-152 = [3, 8, 36, 3]
        width: width multiplier for network (for Wide ResNet)
        bottleneck: adds a bottleneck conv to reduce computation
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    # block 1 (initial conv block)
    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0], width)
        else:
            x = _resnet_block(x, filters[0], width)

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], width, strides=(2, 2))
        else:
            x = _resnet_block(x, filters[k], width, strides=(2, 2))

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k], width)
            else:
                x = _resnet_block(x, filters[k], width)

    x = GroupNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x

def build( config ): 

    image_width = config['image_processing']['image_width']
    image_height = config['image_processing']['image_height']
    image_channels = config['image_processing']['image_channels']

    number_of_classes = config['dataset']['number_of_classes']

    model_file = config['model'].get('model_file', None)
    regularization = config['hyper_parameters'].get('activity_regularizer', None)

    weights = config['model'].get('weights', None)
    print("weights:", weights)
    
    if weights == 'imagenet':
        base_model = ResNet101(
            input_shape=(image_width, image_height, image_channels),
            weights='imagenet',
            include_top=False)
    else:
        base_model = ResNet101(
            input_shape=(image_width, image_height, image_channels),
            weights=None,
            include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if regularization is not None:
        regularization = getattr(importlib.import_module(f'keras.regularizers'), regularization['name'])
        regularization = regularization(**config['hyper_parameters']['activity_regularizer']['params'])
    
    predictions = Dense(
        activity_regularizer=regularization,
        units=number_of_classes,
        activation='softmax',
        name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model