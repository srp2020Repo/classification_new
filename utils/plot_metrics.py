import matplotlib.pyplot as plt

def plot_metric(history, figure_path):

    plt.figure(figsize=(10,5))

    num_epochs = range(1, len(history.history['loss'])+1)
    plt.subplot(1,2,1)
    plt.plot(num_epochs, history.history['loss'], 'bo--')
    plt.plot(num_epochs, history.history['val_loss'], 'ro--')
    plt.title('Training and validation '+ 'loss')
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(["train_"+'loss', 'val_'+'loss'])

    plt.subplot(1,2,1)
    plt.plot(num_epochs, history.history['accuracy'], 'bo--')
    plt.plot(num_epochs, history.history['val_accuracy'], 'ro--')
    plt.title('Training and validation '+ 'Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel('Acc')
    plt.legend(["train_"+'acc', 'val_'+'acc'])
    plt.savefig(figure_path)