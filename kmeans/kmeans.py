from __future__ import print_function

import os
import gzip
import pickle
import math

import numpy as np
import theano
import theano.tensor as T
from PIL import Image
from utils import tile_raster_images

from theano.tensor.shared_randomstreams import RandomStreams

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
        d2 = unpickle('../data/cifar-10-batches-py/data_batch_2')
        d3 = unpickle('../data/cifar-10-batches-py/data_batch_3')


        train_set_x = np.asarray(d['data'])
        #train_set_y = np.asarray(d['labels'])

        train_set_x = np.concatenate((train_set_x, np.asarray(d2['data'])))
        #train_set_y = np.concatenate((train_set_y, np.asarray(d2['labels'])))

        train_set_x = np.concatenate((train_set_x, np.asarray(d3['data'])))
        #train_set_y = np.concatenate((train_set_y, np.asarray(d3['labels'])))

        rval = train_set_x

    return rval



class kmeans(object):

    def __init__(self, X=None, n_centroids=200, e_zka=0.01, e_norm=10):
        if (X is not None):
            #Normalize data
            norm_X= (X - np.mean(X,axis=0)) / np.sqrt((np.var(X,axis=0) + e_norm)) #todo: add whitening

            #Whiten data
            d, V = np.linalg.eig(np.cov(norm_X))
            epsilon = e_zka * np.eye(d.shape[0])
            transformation = np.dot(np.dot(V, np.linalg.inv(np.sqrt(np.diag(d) + epsilon))), V.T)
            self.X=np.dot(transformation, norm_X)
        else:
            print('no data!')
            return 1

        D_init = np.random.randn(self.X.shape[0], n_centroids)
        self.D = theano.shared(value=self.normalizeColumns(D_init), name='D', borrow=True)
        self.S = theano.shared(value=np.zeros((n_centroids,self.X.shape[1])), name='S', borrow=True)

        self.big_pos = T.argmax(abs(T.dot(self.D.T, X)), axis=0)
        #self.updates = [(self.S, self.argmaxColums(T.dot(self.D.T,self.X))),
        self.updates = [(self.S,  T.set_subtensor(T.zeros_like(self.S)[self.big_pos, T.arange(self.big_pos.shape[0])], T.dot(self.D.T, X)[self.big_pos, T.arange(self.big_pos.shape[0])])),
                   (self.D, T.dot(self.X, self.S.T) + self.D)]

        self.train_model = theano.function(
            inputs=[],
            updates=self.updates
        )

        self.normalize_D = theano.function([], updates=[(self.D, self.D / T.sqrt(T.sum(T.sqr(self.D), axis=0)))])
        self.approximation_error = cost = theano.function([], outputs=T.sum(T.sqrt(T.sum(T.sqr(T.dot(self.D, self.S) - X), axis=0))))#(T.dot(self.D, self.S) - self.X).norm(2, axis=0).sum())#




    #def approximation_error(self):
    #    return (T.dot(self.D, self.S) - self.X).norm(2, axis=0).sum()

    def normalizeColumns(self, matrix):
        return matrix / np.linalg.norm(matrix, axis=0, ord=2)

    def normalizeColumnsT(self, matrix):
        return matrix / matrix.norm(2, axis=0)

    #def argmaxColums(self, matrix):
    #    argmaxArray = np.zeros_like(matrix)
    #    ind = T.argmax(matrix, axis=0)
    #    argmaxArray[ind] = 1
    #    return matrix * argmaxArray

def run_kmeans(dataset='cifar-10-python.tar.gz', n_centroids=225, e_zka=0.01, e_norm=10, max_iter=10, n_samples=10000):

    print("...load dataset")
    train_data = load_data(dataset)[:n_samples]



    #only work with greyscale
    #rescale train_data from 32x32 to 12x12
    data_tmp = np.empty((train_data.shape[0], 144))
    for i in xrange(train_data.shape[0]):
        pixels = np.asarray([0.299 *R + 0.587 *G + 0.114*B for (R,G,B) in zip(train_data[i][:1024], train_data[i][1024:2048],train_data[i][2048:3072])]).reshape(32,32)
        image = Image.fromarray(pixels).convert('L')
        image.thumbnail((12, 12))
        data_tmp[i] = np.array(image.getdata(), dtype='uint8')

    train_data = data_tmp


    print("...build model")
    kmean = kmeans(X=train_data.transpose(), n_centroids=n_centroids, e_zka=e_zka, e_norm=e_norm)

    print("...train model")
    for i in range(max_iter):
        kmean.train_model()
        kmean.normalize_D()
        print('Iteration %i: Approximation error: %f' % (i, kmean.approximation_error()))


    image = Image.fromarray(
        tile_raster_images(X=kmean.D.get_value(borrow=True).T,
                           img_shape=(12, 12), tile_shape=(int(math.sqrt(n_centroids)), int(math.sqrt(n_centroids))),
                           tile_spacing=(1, 1)))
    image.save('repflds_white.png')
if __name__ == '__main__':
    run_kmeans()

