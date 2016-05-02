"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

import climin as cli
import climin.initialize as init
import climin.util
import itertools

import matplotlib.pyplot as plt

from logistic_sgd import LogisticRegression, load_data


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
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
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, Weights_1, bias_1, Weights_2, bias_2, activation=T.tanh):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation,
            W=Weights_1,
            b=bias_1
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            W=Weights_2,
            b=bias_2
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input


def relu(x):
    return theano.tensor.switch(x<0, 0, x)

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=500,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500, optimizer='gd', activation=T.tanh):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    tmpl = [(28 * 28, n_hidden), n_hidden, (n_hidden, 10), 10]
    flat, (Weights_1, bias_1, Weights_2, bias_2) = climin.util.empty_with_views(tmpl)

    #cli.initialize.randomize_normal(flat, 0, 1)  # initialize the parameters with random numbers


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


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
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

    gradients = theano.function(
        inputs=[x, y],
        outputs=[
            T.grad(classifier.negative_log_likelihood(y), classifier.hiddenLayer.W),
            T.grad(classifier.negative_log_likelihood(y), classifier.hiddenLayer.b),
            T.grad(classifier.negative_log_likelihood(y), classifier.logRegressionLayer.W),
            T.grad(classifier.negative_log_likelihood(y), classifier.logRegressionLayer.b)
        ],
        allow_input_downcast=True
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = theano.function(
        inputs=[x, y],
        outputs=classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr,
        allow_input_downcast=True
    )
    # end-snippet-4

    def loss(parameters, input, target):
        #Weights, bias = climin.util.shaped_from_flat(parameters, tmpl)

        return cost(input, target)

    def d_loss_wrt_pars(parameters, inputs, targets):
        #Weights, bias = climin.util.shaped_from_flat(parameters, tmpl)

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


    ###############
    # TRAIN MODEL #
    ###############
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
        minibatch_x, minibatch_y = info['args']

        minibatch_avarage_cost = cost(minibatch_x, minibatch_y)
        if iter % validation_frequency == 0:
            # compute zero-one loss on validation set
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

    gd_mlp, gd_losses = test_mlp(optimizer='rprop', n_hidden=300, L1_reg=1.0, L2_reg=0.00, activation=T.tanh, n_epochs=300)

    """
    #Evaluation example for problem 15
    gd_mlp, gd_losses = test_mlp(optimizer='gd', n_hidden=300, L1_reg=1.0, L2_reg=0.00, activation=T.tanh, n_epochs=300)
    gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

    plt.plot(gd_train_loss, '-', linewidth=1, label='train error')
    plt.plot(gd_valid_loss, '-', linewidth=1, label='validation error')
    plt.plot(gd_test_loss, '-', linewidth=1, label='test error')

    plt.legend()
    plt.suptitle('gd-MLP with l1_reg=1.0, l2_reg=0.00, 300 tanh inner neurons')
    plt.savefig('error_gd_adjusted_init_weights_X.png')

    f_repfields, subplot_array = plt.subplots(15, 20)
    weights = gd_mlp.hiddenLayer.W.get_value().transpose()

    for i in range(0,300):
        row = i/20
        column = i % 20

        subplot_array[row][column].imshow(weights[i].reshape((28,28)), cmap = 'Greys_r')
        subplot_array[row][column].axis('off')

    plt.suptitle('gd-MLP with l1_reg=1.0, l2_reg=0.00, 300 tanh inner neurons')
    plt.savefig('repfields_gd_adjusted_init_weights_X.png')

    plt.clf()
    """

    """
    #Evaluation example for problem 16

    gd_mlp, gd_losses = test_mlp(optimizer='gd', n_hidden=300, L1_reg=0.0, L2_reg=1.00, activation=T.nnet.sigmoid, n_epochs=300)
    gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

    plt.plot(gd_train_loss, '-', linewidth=1, label='train error')
    plt.plot(gd_valid_loss, '-', linewidth=1, label='validation error')
    plt.plot(gd_test_loss, '-', linewidth=1, label='test error')

    plt.legend()
    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=1.00, 300 sigmoidal inner neurons')
    plt.savefig('error_sigmoid.png')

    f_repfields, subplot_array = plt.subplots(15, 20)
    weights = gd_mlp.hiddenLayer.W.get_value().transpose()

    for i in range(0, 300):
        row = i / 20
        column = i % 20

        subplot_array[row][column].imshow(weights[i].reshape((28, 28)), cmap='Greys_r')
        subplot_array[row][column].axis('off')

    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=1.00, 300 sigmoidal inner neurons')
    plt.savefig('repfields_sigmoid png')

    plt.clf()

    gd_mlp, gd_losses = test_mlp(optimizer='gd', n_hidden=300, L1_reg=0.0, L2_reg=1.00, activation=relu, n_epochs=300)
    gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

    plt.plot(gd_train_loss, '-', linewidth=1, label='train error')
    plt.plot(gd_valid_loss, '-', linewidth=1, label='validation error')
    plt.plot(gd_test_loss, '-', linewidth=1, label='test error')

    plt.legend()
    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=1.00, 300 rectified linear inner neurons')
    plt.savefig('error_relu.png')

    f_repfields, subplot_array = plt.subplots(15, 20)
    weights = gd_mlp.hiddenLayer.W.get_value().transpose()

    for i in range(0, 300):
        row = i / 20
        column = i % 20

        subplot_array[row][column].imshow(weights[i].reshape((28, 28)), cmap='Greys_r')
        subplot_array[row][column].axis('off')

    plt.suptitle('gd-MLP with l1_reg=0.0, l2_reg=1.00, 300 rectified linear inner neurons')
    plt.savefig('repfields_relu.png')

    plt.clf()
    """