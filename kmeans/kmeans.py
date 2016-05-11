from __future__ import print_function

import math

import numpy as np
import theano
import theano.tensor as T
from PIL import Image
from utils import tile_raster_images
from load_data import load_data


class kmeans(object):

    def __init__(self, X=None, n_centroids=200, e_zka=0.01, e_norm=10):
        if (X is not None):
            #Normalize data
            norm_X= (X - np.mean(X,axis=0)) / np.sqrt((np.var(X,axis=0) + e_norm))

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
        self.updates = [(self.S,  T.set_subtensor(T.zeros_like(self.S)[self.big_pos, T.arange(self.big_pos.shape[0])], T.dot(self.D.T, X)[self.big_pos, T.arange(self.big_pos.shape[0])])),
                   (self.D, T.dot(self.X, self.S.T) + self.D)]

        self.train_model = theano.function(
            inputs=[],
            updates=self.updates
        )

        self.normalize_D = theano.function([], updates=[(self.D, self.D / T.sqrt(T.sum(T.sqr(self.D), axis=0)))])
        self.approximation_error = cost = theano.function([], outputs=T.sum(T.sqrt(T.sum(T.sqr(T.dot(self.D, self.S) - X), axis=0))))

    def normalizeColumns(self, matrix):
        return matrix / np.linalg.norm(matrix, axis=0, ord=2)

    def normalizeColumnsT(self, matrix):
        return matrix / matrix.norm(2, axis=0)

def run_kmeans(dataset='cifar-10-python.tar.gz', n_centroids=225, e_zka=0.01, e_norm=10, max_iter=10, n_samples=10000, grayscale_and_resize=True):

    print("...load dataset")
    (train_data, train_labels) = load_data(dataset)[0]

    #Take number of samples
    train_data = train_data[:n_samples]


    if grayscale_and_resize:
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
    image.save('repfldsX_with_' + str(n_centroids) + 'centroids.png')
if __name__ == '__main__':
    #run_kmeans(n_centroids=100, n_samples=20000)
    #run_kmeans(n_centroids=225, n_samples=20000)
    run_kmeans(n_centroids=484, n_samples=20000)

