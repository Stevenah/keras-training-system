import matplotlib.pyplot as plt

# plot loss and validation loss into graph
def plot_loss(history, file_name):
    
    # setup plots 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    # set plot title
    plt.title('model loss')

    # set axis labels
    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    # setup legend
    plt.legend(['training', 'validation'], loc='upper left')

    # save plot
    plt.savefig(file_name)

    # clear buffer for next plot
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