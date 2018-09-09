from keras.preprocessing.image import ImageDataGenerator

def preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by imagenet mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x


train_data_generator = ImageDataGenerator(
    rescale=1./255)

validation_data_generator = ImageDataGenerator(
    rescale=1./255)