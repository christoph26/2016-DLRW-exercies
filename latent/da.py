from __future__ import print_function

import os
import sys
import timeit

import numpy as np
import math
import climin, climin.util, climin.initialize
import itertools
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from load_data import load_data


try:
    import PIL.Image as Image
except ImportError:
    import Image


class dA(object):

    def __init__(self, numpy_rng, parameters, theano_rng=None, input=None, n_visible=784, n_hidden=500, corruption=0, l1_penalty=0.0):

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.corruption = corruption
        self.l1_penalty = l1_penalty
        self.W = parameters[0]
        self.b = parameters[1]
        self.b_prime = parameters[2]
        self.W_prime = self.W.T
        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

        self.corrupted_input = self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption, dtype=theano.config.floatX) * input
        self.hidden_values = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.reconstructed_values = T.nnet.sigmoid(T.dot(self.hidden_values, self.W_prime) + self.b_prime)
        self.cost = T.mean(- T.sum(self.x * T.log(self.reconstructed_values) + (1 - self.x) * T.log(1 - self.reconstructed_values), axis=1)) + self.l1_penalty * abs(self.hidden_values).sum()
        self.loss = theano.function([self.x], self.cost)
        self.gradients = theano.function([self.x], T.grad(self.cost, self.params))
        self.reconstructed_input = theano.function([input],T.nnet.sigmoid(T.dot(self.hidden_values, self.W_prime) + self.b_prime));

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level, dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

def run_dA(learning_rate=0.1, n_epochs=5, optimizer='gd',
            n_hidden=500, dataset='mnist.pkl.gz', batch_size=20, n_in = 28 * 28, corruption=0.0, l1_penalty=0.0, print_reconstructions=False, print_filters=False):

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.shape[0] // batch_size

    x = T.matrix('x')
    rng = np.random.RandomState(1234)
    theano_rng = RandomStreams(rng.randint(2 ** 30))


    print('...building model')
    dims = [(n_in, n_hidden), n_hidden, n_in]
    flat, (vis_W, hidden_b, vis_b) = climin.util.empty_with_views(dims)

    # initialize with values
    Weights_1_init = rng.uniform(
        low=-4 * np.sqrt(6. / (n_hidden + n_in)),
        high=4 * np.sqrt(6. / (n_hidden + n_in)),
        size=(n_in, n_hidden)
    )

    bias_1_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
    bias_2_init = np.zeros((n_in,), dtype=theano.config.floatX)

    def initialize_in_place(array, values):
        for j in range(0, len(values)):
            array[j] = values[j]

    initialize_in_place(vis_W, Weights_1_init)
    initialize_in_place(hidden_b, bias_1_init)
    initialize_in_place(vis_b, bias_2_init)

    params = [theano.shared(value=vis_W, name='W', borrow=True),
              theano.shared(value=hidden_b, name='b', borrow=True),
              theano.shared(value=vis_b, name='b_prime', borrow=True)]


    da = dA(numpy_rng=rng, parameters=params, theano_rng=theano_rng, input=x, n_visible=28 * 28, n_hidden=500, corruption=corruption, l1_penalty=l1_penalty)

    def d_loss(parameters, inputs, targets):
        g_W, g_hidden_b, g_vis_b = da.gradients(inputs)

        return np.concatenate([g_W.flatten(), g_hidden_b, g_vis_b])
    if not batch_size:
        args = itertools.repeat(([train_set_x, train_set_y], {}))
    else:
        args = ((i, {}) for i in climin.util.iter_minibatches([train_set_x, train_set_y], batch_size, [0, 0]))

    if optimizer == 'gd':
        print('... using gradient descent')
        opt = climin.GradientDescent(flat, d_loss, step_rate=learning_rate, momentum=0.95, args=args)
    elif optimizer == 'rmsprop':
        print('... using rmsprop')
        opt = climin.rmsprop.RmsProp(flat, d_loss, step_rate=0.01, args=args)
    else:
        print('unknown optimizer')
        opt = None

    print('...encoding')
    epoch = 0
    start_time = timeit.default_timer()

    for info in opt:
        iter = info['n_iter']
        if iter % n_train_batches == 1:
            epoch += 1
            this_loss = da.loss(train_set_x)
            print('\nTraining epoch %d, cost ' % epoch, this_loss)
            if epoch >= n_epochs:
                break

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    if print_filters:
        print(('The no corruption code for file ' + os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
        image = Image.fromarray(
            tile_raster_images(X=da.W.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=(int(math.sqrt(n_hidden)), int(math.sqrt(n_hidden))), tile_spacing=(1, 1))
        )
        image.save('filters_'+optimizer+' n_hidden=' + str(n_hidden) + 'corruption=' + str(corruption) + ' and l1_pen='+str(l1_penalty)+'.png', dpi=(300,300))

    if print_reconstructions:
        data = train_set_x[:100]
        reconstruction = da.reconstructed_input(data)
        image = Image.fromarray(
            tile_raster_images(X=reconstruction, img_shape=(28, 28),
                               tile_shape=(10,10), tile_spacing=(1, 1))
        )
        image.save('reconstructions of first 100_' + optimizer + ' n_hidden=' + str(n_hidden) + 'corruption=' + str(
            corruption) + ' and l1_pen=' + str(l1_penalty) + '.png', dpi=(300, 300))




if __name__ == '__main__':
    #run_dA(n_epochs=20,n_hidden=100, corruption=0.0, l1_penalty=0.0, optimizer='rmsprop')
    #run_dA(n_epochs=20, n_hidden=100, corruption=0.3, l1_penalty=0.6, optimizer='rmsprop')
    run_dA(n_epochs=10, n_hidden=1600, corruption=0.3, l1_penalty=0.0, optimizer='rmsprop', print_reconstructions=True)
    run_dA(n_epochs=10, n_hidden=1600, corruption=0.3, l1_penalty=0.3, optimizer='rmsprop', print_reconstructions=True)
    run_dA(n_epochs=10, n_hidden=1600, corruption=0.3, l1_penalty=0.6, optimizer='rmsprop', print_reconstructions=True)


    #run_dA(n_epochs=20, n_hidden=100, corruption=0.3, l1_penalty=0.0, optimizer='gd')
    #run_dA(n_epochs=20, n_hidden=100, corruption=0.3, l1_penalty=0.3, optimizer='gd')
    #run_dA(n_epochs=20, n_hidden=1600, corruption=0.3, l1_penalty=0.0, optimizer='gd')
    #run_dA(n_epochs=20, n_hidden=1600, corruption=0.3, l1_penalty=0.3, optimizer='gd')