import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.utils import shuffle

import six.moves.cPickle as pickle
import gzip
import os


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz' or data_file == 'cifar-10-python.tar.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    if (not os.path.isfile(dataset)) and data_file == 'cifar-10-python.tar.gz':
        from six.moves import urllib
        origin = (
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

        import tarfile
        tar = tarfile.open(dataset)
        tar.extractall(path='../data')
        tar.close()

    print('... loading data')

    # Load the dataset
    if data_file == 'mnist.pkl.gz':
        with gzip.open(dataset, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
    else:

        def unpickle(file):
            fo = open(file, 'rb')
            dict = pickle.load(fo)
            fo.close()
            return dict

        d = unpickle('../data/cifar-10-batches-py/data_batch_1')
        d2 = unpickle('../data/cifar-10-batches-py/data_batch_1')
        d3 = unpickle('../data/cifar-10-batches-py/data_batch_1')


        train_set_x = np.asarray(d['data'])
        train_set_y = np.asarray(d['labels'])

        train_set_x = np.concatenate((train_set_x, np.asarray(d2['data'])))
        train_set_y = np.concatenate((train_set_y, np.asarray(d2['labels'])))

        train_set_x = np.concatenate((train_set_x, np.asarray(d3['data'])))
        train_set_y = np.concatenate((train_set_y, np.asarray(d3['labels'])))

    rval = [(train_set_x, train_set_y)]

    return rval

def doPCA(data, n_components=2):

    data2 = data - np.mean(data, axis=0)
    covarance = np.cov(data2.transpose())

    W, V = np.linalg.eig(covarance)
    #zipped = zip(W,V)
    #zipped.sort()
    #zipped = zip(V,np.hsplit(W,np.arange(1,len(W))))
    zipped = zip(W, V.transpose())
    rearrangedEvalsVecs = sorted(zipped,key=lambda x: x[0].real, reverse=True)

    W,V = zip(*rearrangedEvalsVecs)
    print('... pca performed')

    return np.asarray(V).transpose()[:,:n_components]
    #truncated = rearrangedEvalsVecs[0][0]

    #for i in range(0, n_components):
    #    truncated = np.concatenate((truncated, rearrangedEvalsVecs[i][0]))

    #truncated = np.concatenate((rearrangedEvalsVecs[i][0] for i in range(n_components)))

    #return V[:,:n_components]
    #return np.reshape(truncated, (n_components,-1)).transpose()


def doPCAScatterplot(dataset='mnist.pkl.gz', n_samples=30000):
    datasets = load_data(dataset)

    X_train, y_train = datasets[0]

    X_train, y_train = shuffle(X_train, y_train)
    X_train, y_train = X_train[:n_samples], y_train[:n_samples]  # lets subsample a bit for a first impression

    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()

    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue


        X_ = X_train[(y_train == i) + (y_train == j)]
        y_ = y_train[(y_train == i) + (y_train == j)]

        PCA= doPCA(X_)
        X_transformed = np.dot(X_, PCA)
        plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_, marker='o',edgecolor='none')
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())

        plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_, marker='o',edgecolor='none')
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            plots[i, j].set_title(j)
            plots[j, i].set_ylabel(j)

    plt.tight_layout()
    #plt.savefig("scatterplotCIFAR.png")
    plt.savefig("scatterplotNMIST.png")

#doPCAScatterplot('cifar-10-python.tar.gz')
doPCAScatterplot('mnist.pkl.gz')