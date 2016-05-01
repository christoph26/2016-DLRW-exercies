#Multi Layer Perceptron

##Problem 14
The file mlp.py contains an implementation of a neuronal network with one hidden layer, early stopping, regularization and mini-batch minimization with climin for the NMIST dataset. The implementation depends on the file logistic_sgd.py which contains a multiclass logistic regression class analogous the the previous exercise.

Details for an execution can be specified with the parameters of the method test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=500, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500, optimizer='gd', activation=T.tanh).

learning_rate: Learing rate for optimization with gradient descent
l1_reg, l2_reg: factors for l1 and l2 regularization
n_hidden: number neurons in the hidden layer,
optimizer: name of the climin optimizer, which should be used. Possiblilties are "gd", "rprop", "rmsprop", etc.
activation: activation function for the hidden layer.

##Problem 15

For such a neuronal network it is crucial to initialize the weights correctly. With and normal inizialization of all parameters with the call

cli.initialize.randomize_normal(flat, 0, 1)

the neuronal network cannot learn properly and only archieves validation and test errors of over 5%. However, if the weights are initialized with an uniformal distribution as described in the tutorial, the MLP performs greatly achieveing error rates of less than 1.8 already after 20 epochs, test and validation rates of 1.690000 % after 53 epochs and

The following table shows a few result of classifications with 300 hidden neurons and tanh as activation function for varying regularizations. Respective error plots and visualizations of the weights are in the subfolder "Visualizations Problem 15". It also contains an error pot of a run with bad weight initialization as described above.


| l1_reg  | l2_reg | 1. run: best validation score | 1. run: test error| 2. run: best validation score | 2. run: test error |
|---------|--------|-------------------------------|-------------------|-------------------------------|--------------------|
| 0.001   | 0.0001 | 1,740000 %                    | 1,840000 %        |                               |                    |
| 0.1     | 0.01   | 1,780000 %                    | 1,820000 %        |                               |                    |
| 0.0     | 1.0    | 1.610000 %                    | 1.730000 %        | 1.640000 %                    | 1.730000 %         |
| 1.0     | 0.0    | 1.830000 %                    | 1.780000 %        | 1.710000 %                    | 1.830000 %         |
| 1.0     | 1.0    | 1.660000 %                    | 1.750000 %        | 1.790000 %                    | 1.730000 %         |


Best validation score of  with test performance 1.730000 %

Optimization complete. Best validation score of 1.830000 % with test performance 1.780000 %
Optimization complete. Best validation score of 1.660000 % with test performance 1.750000 %

The code for file mlp.py ran for 71.97m
epoch 300, minibatch 1/2500, validation error  1.720000 %, iter/patience 750000/1145000
Optimization complete. Best validation score of 1.710000 % with test performance 1.830000 %

epoch 110, minibatch 1/2500, validation error  1.820000 %, iter/patience 275000/275000
Optimization complete. Best validation score of 1.790000 % with test performance 1.730000 %
The code for file mlp.py ran for 28.28m

The code for file mlp.py ran for 7.29m
epoch 30, minibatch 1/2500, validation error  1.730000 %, iter/patience 75000/75000
Optimization complete. Best validation score of 1.640000 % with test performance 1.730000 %


#Problem 16




##Problem 17
The files error_tanh.png, error_sigmoid.png, and error_relu.png contain error plots of execution with different activation functions.

##Problem 18
The files repflds_tanh.png, repflds_sigmoid.png, and repflds_relu.png contain visualizations of the weights of the 300 hidden units of executions with tanh, sigmoid and rectified liean neurons.

##Problem 19

By initializing the weights optimal, the achieved error rate can be reduced to less than 2%. With regularization, the error rates even drop under 1.7%. Concrete executions with such results are given above in the capter of problem 15.