from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

import climin as cli
import climin.util
import itertools

import matplotlib.pyplot as plt

from logistic_sgd import LogisticRegression, load_data


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out, Weights_1, bias_1, Weights_2, bias_2, activation=T.tanh):

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation,
            W=Weights_1,
            b=bias_1
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            W=Weights_2,
            b=bias_2
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input


def relu(x):
    return theano.tensor.switch(x<0, 0, x)

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=500,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500, optimizer='gd', activation=T.tanh):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    tmpl = [(28 * 28, n_hidden), n_hidden, (n_hidden, 10), 10]
    flat, (Weights_1, bias_1, Weights_2, bias_2) = climin.util.empty_with_views(tmpl)

    #Initialize weights with uniformal distribution according to the tutorial
    rng = numpy.random.RandomState(1234)
    Weights_1_init = rng.uniform(
        low=-numpy.sqrt(6. / (28*28 + n_hidden)),
        high=numpy.sqrt(6. / (28*28 + n_hidden)),
        size=(28*28, n_hidden)
    )

    Weights_2_init = rng.uniform(
        low=-numpy.sqrt(6. / (n_hidden+10)),
        high=numpy.sqrt(6. / (n_hidden+10)),
        size=(n_hidden, 10)
    )

    bias_1_init = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
    bias_2_init = numpy.zeros((10,), dtype=theano.config.floatX)

    if activation == T.nnet.sigmoid:
        Weights_1_init *= 4
        Weights_2_init *= 4

    def initialize_in_place(array, values):
        for j in range(0, len(values)):
            array[j] = values[j]

    initialize_in_place(Weights_1, Weights_1_init)
    initialize_in_place(Weights_2, Weights_2_init)
    initialize_in_place(bias_1, bias_1_init)
    initialize_in_place(bias_2, bias_2_init)


    if batch_size is None:
        args = itertools.repeat(([train_set_x, train_set_y], {}))
        n_train_batches = 1
    else:
        args = cli.util.iter_minibatches([train_set_x, train_set_y], batch_size, [0, 0])
        args = ((i, {}) for i in args)
        n_train_batches = train_set_x.shape[0] // batch_size


    print('... building the model')

    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        Weights_1=theano.shared(value = Weights_1, name = 'W', borrow = True),
        bias_1=theano.shared(value = bias_1, name = 'b', borrow = True),
        Weights_2=theano.shared(value = Weights_2, name = 'W', borrow = True),
        bias_2=theano.shared(value = bias_2, name = 'b', borrow = True),
        activation=T.tanh
    )

    #cost with regularisation terms
    cost = theano.function(
        inputs=[x, y],
        outputs=classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr,
        allow_input_downcast=True
    )

    # gradients with regularisation terms
    gradients = theano.function(
        inputs=[x, y],
        outputs=[
            T.grad(classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr, classifier.hiddenLayer.W),
            T.grad(classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr, classifier.hiddenLayer.b),
            T.grad(classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr, classifier.logRegressionLayer.W),
            T.grad(classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr, classifier.logRegressionLayer.b)
        ],
        allow_input_downcast=True
    )



    def loss(parameters, input, target):
        return cost(input, target)

    def d_loss_wrt_pars(parameters, inputs, targets):
        g_W_1, g_b_1, g_W_2, g_b_2 = gradients(inputs, targets)

        return numpy.concatenate([g_W_1.flatten(), g_b_1, g_W_2.flatten(), g_b_2])

    zero_one_loss = theano.function(
        inputs=[x, y],
        outputs=classifier.errors(y),
        allow_input_downcast=True
    )

    if optimizer == 'gd':
        print('... using gradient descent')
        opt = cli.GradientDescent(flat, d_loss_wrt_pars, step_rate=learning_rate, momentum=.95, args=args)
    elif optimizer == 'bfgs':
        print('... using using quasi-newton BFGS')
        opt = cli.Bfgs(flat, loss, d_loss_wrt_pars, args=args)
    elif optimizer == 'lbfgs':
        print('... using using quasi-newton L-BFGS')
        opt = cli.Lbfgs(flat, loss, d_loss_wrt_pars, args=args)
    elif optimizer == 'nlcg':
        print('... using using non linear conjugate gradient')
        opt = cli.NonlinearConjugateGradient(flat, loss, d_loss_wrt_pars, min_grad=1e-03, args=args)
    elif optimizer == 'rmsprop':
        print('... using rmsprop')
        opt = cli.RmsProp(flat, d_loss_wrt_pars, step_rate=1e-4, decay=0.9, args=args)
    elif optimizer == 'rprop':
        print('... using resilient propagation')
        opt = cli.Rprop(flat, d_loss_wrt_pars, args=args)
    elif optimizer == 'adam':
        print('... using adaptive momentum estimation optimizer')
        opt = cli.Adam(flat, d_loss_wrt_pars, step_rate=0.0002, decay=0.99999999, decay_mom1=0.1, decay_mom2=0.001,
                       momentum=0, offset=1e-08, args=args)
    elif optimizer == 'adadelta':
        print('... using adadelta')
        opt = cli.Adadelta(flat, d_loss_wrt_pars, step_rate=1, decay=0.9, momentum=.95, offset=0.0001, args=args)
    else:
        print('unknown optimizer')
        return 1

    print('... training')

    # early stopping parameters
    if batch_size == None:
        patience = 250
    else:
        patience = 10000  # look at this many samples regardless

    patience_increase = 2  # wait this mutch longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this mutch is considered signigicant
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    test_loss = 0.

    valid_losses = []
    train_losses = []
    test_losses = []

    epoch = 0

    start_time = timeit.default_timer()

    for info in opt:
        iter = info['n_iter']
        epoch = iter // n_train_batches
        minibatch_index = iter % n_train_batches

        if iter % validation_frequency == 0:
            validation_loss = zero_one_loss(valid_set_x, valid_set_y)
            valid_losses.append(validation_loss)
            train_losses.append(zero_one_loss(train_set_x, train_set_y))
            test_losses.append(zero_one_loss(test_set_x, test_set_y))

            print(
                'epoch %i, minibatch %i/%i, validation error % f %%, iter/patience %i/%i' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    validation_loss * 100,
                    iter,
                    patience
                )
            )
            # if we got the best validation score until now
            if validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
                if validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iter * patience_increase)
                best_validation_loss = validation_loss
                # test it on the test set
                test_loss = zero_one_loss(test_set_x, test_set_y)

                print(
                    '    epoch %i, minibatch %i/%i, test error of best model %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_loss * 100
                    )
                )


        if patience <= iter or epoch >= n_epochs:
            break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% with test performance %f %%') %
          (best_validation_loss * 100., test_loss * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    losses = (train_losses, valid_losses, test_losses)

    return classifier, losses

if __name__ == '__main__':

    #gd_mlp, gd_losses = test_mlp(optimizer='gd', n_hidden=300, L1_reg=0.0, L2_reg=0.00, activation=T.tanh, n_epochs=300)

    """
    #Evaluation example for problem 15
    gd_mlp, gd_losses = test_mlp(optimizer='gd', n_hidden=300, L1_reg=0.0, L2_reg=0.00, activation=T.tanh, n_epochs=300)
    gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

    plt.plot(gd_train_loss, '-', linewidth=1, label='train error')
    plt.plot(gd_valid_loss, '-', linewidth=1, label='validation error')
    plt.plot(gd_test_loss, '-', linewidth=1, label='test error')

    plt.legend()
    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=0.00, 300 tanh inner neurons')
    plt.savefig('error_gd_adjusted_init_weights_1.png')

    f_repfields, subplot_array = plt.subplots(15, 20)
    weights = gd_mlp.hiddenLayer.W.get_value().transpose()

    for i in range(0,300):
        row = i/20
        column = i % 20

        subplot_array[row][column].imshow(weights[i].reshape((28,28)), cmap = 'Greys_r')
        subplot_array[row][column].axis('off')

    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=0.00, 300 tanh inner neurons')
    plt.savefig('repfields_gd_adjusted_init_weights_1.png')

    plt.clf()
    """

    """
    #Evaluation example for problem 16

    gd_mlp, gd_losses = test_mlp(optimizer='gd', n_hidden=300, L1_reg=0.0, L2_reg=0.00001, activation=T.nnet.sigmoid, n_epochs=300)
    gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

    plt.plot(gd_train_loss, '-', linewidth=1, label='train error')
    plt.plot(gd_valid_loss, '-', linewidth=1, label='validation error')
    plt.plot(gd_test_loss, '-', linewidth=1, label='test error')

    plt.legend()
    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=0.00001, 300 sigmoidal inner neurons')
    plt.savefig('error_sigmoid.png')

    f_repfields, subplot_array = plt.subplots(15, 20)
    weights = gd_mlp.hiddenLayer.W.get_value().transpose()

    for i in range(0, 300):
        row = i / 20
        column = i % 20

        subplot_array[row][column].imshow(weights[i].reshape((28, 28)), cmap='Greys_r')
        subplot_array[row][column].axis('off')

    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=0.00001, 300 sigmoidal inner neurons')
    plt.savefig('repfields_sigmoid.png')

    plt.clf()

    gd_mlp, gd_losses = test_mlp(optimizer='gd', n_hidden=300, L1_reg=0.0, L2_reg=0.00001, activation=relu, n_epochs=300)
    gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

    plt.plot(gd_train_loss, '-', linewidth=1, label='train error')
    plt.plot(gd_valid_loss, '-', linewidth=1, label='validation error')
    plt.plot(gd_test_loss, '-', linewidth=1, label='test error')

    plt.legend()
    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=0.00001, 300 rectified linear inner neurons')
    plt.savefig('error_relu.png')

    f_repfields, subplot_array = plt.subplots(15, 20)
    weights = gd_mlp.hiddenLayer.W.get_value().transpose()

    for i in range(0, 300):
        row = i / 20
        column = i % 20

        subplot_array[row][column].imshow(weights[i].reshape((28, 28)), cmap='Greys_r')
        subplot_array[row][column].axis('off')

    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=0.00001, 300 rectified linear inner neurons')
    plt.savefig('repfields_relu.png')

    plt.clf()
    """