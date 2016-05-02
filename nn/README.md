#Multi Layer Perceptron

##Problem 14
The file mlp.py contains an implementation of a neuronal network with one hidden layer, early stopping, regularization and mini-batch minimization with climin for the NMIST dataset. The implementation depends on the file logistic_sgd.py which contains a multiclass logistic regression class analogous the the previous exercise.

Details for an execution can be specified with the parameters of the method test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=500, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500, optimizer='gd', activation=T.tanh).

learning_rate: Learing rate for optimization with gradient descent
l1_reg, l2_reg: factors for l1 and l2 regularization
n_hidden: number neurons in the hidden layer,
optimizer: name of the climin optimizer, which should be used. Possiblilties are "gd", "rprop", "rmsprop", etc.
activation: activation function for the hidden layer.
batch_size: batch size
n_epochs: maximal number of epochs.


##Problem 15

For such a neuronal network it is crucial to initialize the weights correctly. With and normal inizialization of all parameters with the call

cli.initialize.randomize_normal(flat, 0, 1)

the neuronal network cannot learn properly and only archieves validation and test errors of over 5%. An error plot of such a run is in the subfolder "Visualizations Problem 15".

However, if the weights are initialized with an uniformal distribution as described in the tutorial, the MLP performs greatly achieveing error rates of less than 1.8 already after 20 epochs, and final test and validation rates of 1.6% - 1.7%. The following table shows a few result of classifications with 300 hidden neurons, gradient descent and tanh as activation function for varying regularizations. Respective error plots and visualizations of the weights are in the subfolder "Visualizations Problem 15".


| l1_reg  | l2_reg | 1. run: best validation score | 1. run: test error| 2. run: best validation score | 2. run: test error |
|---------|--------|-------------------------------|-------------------|-------------------------------|--------------------|
| 0.001   | 0.0001 | 1,740000 %                    | 1,840000 %        |                               |                    |
| 0.1     | 0.01   | 1,780000 %                    | 1,820000 %        |                               |                    |
| 0.0     | 1.0    | 1.610000 %                    | 1.730000 %        | 1.640000 %                    | 1.730000 %         |
| 1.0     | 0.0    | 1.830000 %                    | 1.780000 %        | 1.710000 %                    | 1.830000 %         |
| 1.0     | 1.0    | 1.660000 %                    | 1.750000 %        | 1.790000 %                    | 1.730000 %         |

The runs show, that the MLP-classifications works better with regularization. Whereas runs with no or less regularization produced test errors of 1.8, runs with stronger regularization could achieve better test errors. a second observation is that in this szenario l2 regularization lead to better improvements compared to l1 regularization. The optimal value in these executions was 1.73% and could only be achieved by runs with l2_reg=1.0. 

Furthermore it can be seen, that the runtime is highly random due to the randomized mini-batches. The first run with l1_reg= 0.0 and l2-reg=1.0 for example took 500 epochs. Here the validation error was constantly improved in a way that the early stopping mechanism was not activated. In contrast, the second run with the same parameters only took 30 epochs producing the same test error score of 1.73. Respective plots of the two executions are in the subfolder "Visualizations Problem 15" in the files "error_gd_adjusted_init_weights_3.png" and "error_gd_adjusted_init_weights_6.png".

As a consequence a further optimization could be to improve the stopping creteria. For the setting of this exercise the major minimization is done in the first few epochs. An alternative could therefore be, to skip early stopping and set the maximal number of epochs to 50.


#Problem 16

The following evaluations use gradient descent, 300 activation function and l2 regulation with a factor 1.0.

The weight initialization for the tanh and sigmoid functions is done according to Bengio, X. Glorot, Understanding the difficulty of training deep feedforward neuralnetworks, AISTATS 2010. Being similarly shaped as a tanh, the retified linear unit weights are initialized the same as in the run with tanh activation functions.

| activation function    | best validation score         | test error        |
|------------------------|-------------------------------|-------------------|
| tanh                   | 1.640000 %                    | 1.730000 %        |
| sigmoid                | 2.080000 %                    | 2.080000 %        |
| rectified linear units | 1.720000 %                    | 1.770000 %        |

Comparing the results shows that tanh activation functions produce the best results. Considering the shape of the different functions, this obervation can be explained. Tanh has an image of [-1,1] and is flater than the sigmoid function. The sigmoid only outputs values in the interval [0,1] and offers therefore a less "powerfull" activations for a MLP. The rectified linear units also have no negative image, but are similarly shaped as the tanh. Their perfomance is not as great as the tanh activation functions, but notably better than sigmoid functions.

A crucial factor is, that sigmoid functions only output 0 for x-> -infinity. Since high weights in the hidden layer are necessary to generate inputs for the activation functions with high modulus, it is more difficult for single neurons to specialize on a single feature in the classification progress. The tanh and the rectified linear units already output zero for an input of zero and thus encourage small weights.

This can also be observed in the plots of the respective fields. The weight visualizations of tanh and rectified linear units contain one core black area, which represents the feature this neuron focuses on. The rest of the weights are quite homogenious grey. The plot of the sigmoid function in contrast contains less clearer shape. Instead the weights of all neurons seem similar and have a dotted background pattern.

##Problem 17
The files error_tanh.png, error_sigmoid.png, and error_relu.png contain error plots of execution with different activation functions.

##Problem 18
The files repflds_tanh.png, repflds_sigmoid.png, and repflds_relu.png contain visualizations of the weights of the 300 hidden units of executions with tanh, sigmoid and rectified liean neurons.

##Problem 19

By initializing the weights optimal, the achieved error rate can be reduced to less than 2%. With regularization, the error rates even drop under 1.7%. Concrete executions with such results are given above in the capter of problem 15.