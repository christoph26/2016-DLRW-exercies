from __future__ import print_function

import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

import climin as cli
import climin.initialize as init
import climin.util
import itertools

from load_data import load_data

import matplotlib.pyplot as plt

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, W = None, b = None):

        if W is None:
            W_values = np.zeros((n_in, n_out), dtype = theano.config.floatX)
            W = theano.shared(value = W_values, name = 'W', borrow = True )

        if b is None:
            b_values = np.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True )

        self.W = W
        self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def sgd_optimization_mnist(learning_rate=0.01, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600, optimizer='gd'):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    tmpl = [(28 * 28, 10), 10]
    flat, (Weights, bias) = climin.util.empty_with_views(tmpl)

    cli.initialize.randomize_normal(flat, 0, 1)

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

    classifier = LogisticRegression(
            input = x,
            n_in = 28 * 28,
            n_out = 10,
            W = theano.shared(value = Weights, name = 'W', borrow = True),
            b = theano.shared(value = bias, name = 'b', borrow = True)
            )

    gradients = theano.function(
            inputs = [x, y],
            outputs = [
                T.grad(classifier.negative_log_likelihood(y), classifier.W),
                T.grad(classifier.negative_log_likelihood(y), classifier.b)
                ],
            allow_input_downcast = True
            )

    cost = theano.function(
        inputs=[x, y],
        outputs=classifier.negative_log_likelihood(y),
        allow_input_downcast=True
    )

    def loss(parameters, input, target):
        return cost(input, target)

    def d_loss_wrt_pars(parameters, inputs, targets):
        g_W, g_b = gradients(inputs, targets)

        return np.concatenate([g_W.flatten(), g_b])

    zero_one_loss = theano.function(
            inputs = [x, y],
            outputs = classifier.errors(y),
            allow_input_downcast = True
            )

    if optimizer == 'gd':
        print('... using gradient descent')
        opt = cli.GradientDescent(flat, d_loss_wrt_pars, step_rate=learning_rate, momentum=.95, args=args)
    elif optimizer == 'rmsprop':
        print('... using rmsprop')
        opt = cli.RmsProp(flat, d_loss_wrt_pars, step_rate=1e-4, decay=0.9, args=args)
    elif optimizer == 'rprop':
        print('... using resilient propagation')
        opt = cli.Rprop(flat, d_loss_wrt_pars, args=args)
    elif optimizer == 'adam':
        print('... using adaptive momentum estimation optimizer')
        opt = cli.Adam(flat, d_loss_wrt_pars, step_rate = 0.0002, decay = 0.99999999, decay_mom1 = 0.1, decay_mom2 = 0.001, momentum = 0, offset = 1e-08, args=args)
    elif optimizer == 'adadelta':
        print('... using adadelta')
        opt = cli.Adadelta(flat, d_loss_wrt_pars, step_rate=1, decay = 0.9, momentum = .95, offset = 0.0001, args=args)
    else:
        print('unknown optimizer')
        return 1

    print('... training the model')

    # early stopping parameters
    if batch_size== None:
        patience = 250
    else:
        patience = 5000 # look at this many samples regardless

    patience_increase = 2 # wait this mutch longer when a new best is found
    improvement_threshold = 0.995 # a relative improvement of this mutch is considered signigicant
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = np.inf
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

    print('Optimization complete with best validation score of %f %%, with test performance %f %%' % (best_validation_loss * 100., test_loss * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    losses = (train_losses, valid_losses, test_losses)

    return classifier, losses

if __name__ == "__main__":

    plot_learning_rate_x = []
    plot_learning_rate =[]

    """
    # Plot for varying learning rate of gradient descent

    for i in range(1,20,1):
        learning_rate= i/20.0
        gd_classifier, gd_losses = sgd_optimization_mnist(learning_rate=learning_rate, optimizer='gd')
        gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

        plot_learning_rate_x.append(learning_rate)
        plot_learning_rate.append(gd_test_loss[len(gd_test_loss)-1])

    plt.plot(plot_learning_rate_x,plot_learning_rate, '-', linewidth = 1, label = 'rest error')
    plt.legend()
    plt.savefig('learning_rate.png')
    """

    """
    # Plot of variation due to randomness in the minibatch selection

    for i in range(0,10,1):
        gd_classifier, gd_losses = sgd_optimization_mnist(learning_rate=0.55, optimizer='gd')
        gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

        plot_learning_rate.append(gd_test_loss[len(gd_test_loss)-1])

    plt.plot(plot_learning_rate, '-', linewidth = 1, label = 'test error')
    plt.legend()
    plt.savefig('Varying_with_same_parameter.png')
    """

    gd_classifier, gd_losses = sgd_optimization_mnist(optimizer='gd')
    gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

    """
    # Plot of the error rates of gradient descent.

    plt.plot(gd_train_loss, '-', linewidth = 1, label = 'train error')
    plt.plot(gd_valid_loss, '-', linewidth = 1, label = 'validation error')
    plt.plot(gd_test_loss, '-', linewidth = 1, label = 'test error')


    plt.legend()
    plt.savefig('error_gd.png')
    """

    """
    # Plot of the error rates of different minimization methods

    #nlcg_classifier, nlcg_losses = sgd_optimization_mnist(optimizer='nlcg')
    #nlcg_train_loss, nlcg_valid_loss, nlcg_test_loss = nlcg_losses

    rms_classifier, rms_losses = sgd_optimization_mnist(optimizer='rmsprop')
    rms_train_loss, rms_valid_loss, rms_test_loss = rms_losses

    rprop_classifier, rprop_losses = sgd_optimization_mnist(optimizer='rprop')
    rprop_train_loss, rprop_valid_loss, rprop_test_loss = rprop_losses

    adam_classifier, adam_losses = sgd_optimization_mnist(optimizer='adam')
    adam_train_loss, adam_valid_loss, adam_test_loss = adam_losses

    adadelta_classifier, adadelta_losses = sgd_optimization_mnist(optimizer='adadelta')
    adadelta_train_loss, adadelta_valid_loss, adadelta_test_loss = adadelta_losses

    f_errors, (gd_plt, rms_plt, rprop_plt, adam_plt, adadelta_plt) = plt.subplots(5)

    gd_plt.plot(gd_train_loss, '-', linewidth=1, label='train loss')
    gd_plt.plot(gd_valid_loss, '-', linewidth=1, label='vaidation loss')
    gd_plt.plot(gd_test_loss, '-', linewidth=1, label='test loss')
    gd_plt.set_title('gd')

    #lbfgs_plt.plot(lbfgs_train_loss, '-', linewidth=1, label='train loss')
    #lbfgs_plt.plot(lbfgs_valid_loss, '-', linewidth=1, label='vaidation loss')
    #lbfgs_plt.plot(lbfgs_test_loss, '-', linewidth=1, label='test loss')

    #nlcg_plt.plot(nlcg_train_loss, '-', linewidth=1, label='train loss')
    #nlcg_plt.plot(nlcg_valid_loss, '-', linewidth=1, label='vaidation loss')
    #nlcg_plt.plot(nlcg_test_loss, '-', linewidth=1, label='test loss')
    #nlcg_plt.set_title('nlcg')

    rms_plt.plot(rms_train_loss, '-', linewidth=1, label='train loss')
    rms_plt.plot(rms_valid_loss, '-', linewidth=1, label='vaidation loss')
    rms_plt.plot(rms_test_loss, '-', linewidth=1, label='test loss')
    rms_plt.set_title('rms')

    rprop_plt.plot(rprop_train_loss, '-', linewidth=1, label='train loss')
    rprop_plt.plot(rprop_valid_loss, '-', linewidth=1, label='vaidation loss')
    rprop_plt.plot(rprop_test_loss, '-', linewidth=1, label='test loss')
    rprop_plt.set_title('rprop')

    adam_plt.plot(adam_train_loss, '-', linewidth=1, label='train loss')
    adam_plt.plot(adam_valid_loss, '-', linewidth=1, label='vaidation loss')
    adam_plt.plot(adam_test_loss, '-', linewidth=1, label='test loss')
    adam_plt.set_title('adam')

    adadelta_plt.plot(adadelta_train_loss, '-', linewidth=1, label='train loss')
    adadelta_plt.plot(adadelta_valid_loss, '-', linewidth=1, label='vaidation loss')
    adadelta_plt.plot(adadelta_test_loss, '-', linewidth=1, label='test loss')
    adadelta_plt.set_title('adadelta')

    #red_patch = mpatches.Patch(color='red', label='Test error')
    #green_patch = mpatches.Patch(color='green', label='Validation error')
    #blue_patch = mpatches.Patch(color='blue', label='Train error')
    #plt.legend(ncol=3, handles=[red_patch, green_patch, blue_patch])

    #plt.legend()
    #plt.legend(loc=0, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('errors.png')
    """

    """"
    # Visualize the final wights as images (one for each digit)

    f_repfields, ((zero, one, two, three, four), (five, six, seven, eight, nine)) = plt.subplots(2,5)

    weights = gd_classifier.W.get_value().transpose()
    zero.imshow(weights[0].reshape((28,28)), cmap = 'Greys_r')
    zero.set_title('zero')
    one.imshow(weights[1].reshape((28, 28)), cmap='Greys_r')
    one.set_title('one')
    two.imshow(weights[2].reshape((28, 28)), cmap='Greys_r')
    two.set_title('two')
    three.imshow(weights[3].reshape((28, 28)), cmap='Greys_r')
    three.set_title('three')
    four.imshow(weights[4].reshape((28, 28)), cmap='Greys_r')
    four.set_title('four')
    five.imshow(weights[5].reshape((28, 28)), cmap='Greys_r')
    five.set_title('five')
    six.imshow(weights[6].reshape((28, 28)), cmap='Greys_r')
    six.set_title('six')
    seven.imshow(weights[7].reshape((28, 28)), cmap='Greys_r')
    seven.set_title('seven')
    eight.imshow(weights[8].reshape((28,28)), cmap = 'Greys_r')
    eight.set_title('eight')
    nine.imshow(weights[9].reshape((28, 28)), cmap='Greys_r')
    nine.set_title('nine')

    zero.axis('off')
    one.axis('off')
    two.axis('off')
    three.axis('off')
    four.axis('off')
    five.axis('off')
    six.axis('off')
    seven.axis('off')
    eight.axis('off')
    nine.axis('off')

    plt.savefig('repfields.png')
    """
