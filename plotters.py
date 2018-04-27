import matplotlib.pyplot as plt

# plot loss and validation loss into graph
def plot_loss(history, file_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(file_name)

    plt.gcf().clear()

# plot accuracy and validation accuracyinto graph
def plot_accuracy(history, file_name):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(file_name)

    plt.gcf().clear()