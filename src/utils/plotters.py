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
    
    # plot accuracy
    plt.plot(history.history['acc'])
    
    # plot validation accuracy
    plt.plot(history.history['val_acc'])

    # set plot title
    plt.title('model accuracy')

    # set y label to 'accuracy'
    plt.ylabel('accuracy')

    # set x label to 'epoch'
    plt.xlabel('epoch')

    # add legend to plot
    plt.legend(['training', 'validation'], loc='upper left')

    # save plot to disk
    plt.savefig(file_name)

    # clear plot
    plt.gcf().clear()