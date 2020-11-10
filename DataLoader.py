from mlxtend.data import loadlocal_mnist
import platform
import math

class DataLoader:
    def __init__(self):
        self.train = []
        self.val = []
        self.test = []
        
    def load_trainval(self):
        # load and flatten data
        if not platform.system() == 'Windows':
            X, y = loadlocal_mnist(
                    images_path='train-images-idx3-ubyte', 
                    labels_path='train-labels-idx1-ubyte')

        else:
            X, y = loadlocal_mnist(
                    images_path='train-images-idx3-ubyte', 
                    labels_path='train-labels-idx1-ubyte')

        # set cutoff between train and validation data
        cutoff = math.floor((2/3)*X.shape[0])

        # set train and validation data
        self.train = {
            'data': X[:cutoff],
            'label': y[:cutoff]
        }
        self.val = {
            'data': X[cutoff:],
            'label': y[cutoff:]
        }

    def load_test(self):
        # load and flatten data
        if not platform.system() == 'Windows':
            X, y = loadlocal_mnist(
                    images_path='t10k-images-idx3-ubyte', 
                    labels_path='t10k-labels-idx1-ubyte')

        else:
            X, y = loadlocal_mnist(
                    images_path='t10k-images-idx3-ubyte', 
                    labels_path='t10k-labels-idx1-ubyte')

        # set test data
        self.test = {
            'data': X,
            'label': y
        }

    def get_train(self):
        return self.train

    def get_val(self):
        return self.val

    def get_test(self):
        return self.test
