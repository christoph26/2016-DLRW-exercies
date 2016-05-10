import numpy as np

import six.moves.cPickle as pickle
import gzip
import os


def load_data(dataset):

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
        d2 = unpickle('../data/cifar-10-batches-py/data_batch_2')
        d3 = unpickle('../data/cifar-10-batches-py/data_batch_3')

        train_set_x = np.asarray(d['data'])
        train_set_y = np.asarray(d['labels'])

        train_set_x = np.concatenate((train_set_x, np.asarray(d2['data'])))
        train_set_y = np.concatenate((train_set_y, np.asarray(d2['labels'])))

        train_set_x = np.concatenate((train_set_x, np.asarray(d3['data'])))
        train_set_y = np.concatenate((train_set_y, np.asarray(d3['labels'])))

        rval = [(train_set_x, train_set_y)]

    return rval
