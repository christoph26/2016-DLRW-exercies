#PCA and sparse Autoencoder
##Problem 20

The file "pca.py" contains an implementation of pca. The actual pca is done in the method doPCA(data, n_components=2). Data is an array with the for the PCA, n_components is the number of eigenvectors, which should be returned. the returned array can be used to truncate orginal data.

##Problem 21
The method doPCAScatterplot(dataset='mnist.pkl.gz', n_samples=30000) performs the scatterplot. dataset specifies with 'mnist.pkl.gz' or 'cifar-10-python.tar.gz' the dataset, n_samples the number of randomly select samples for the plot. The current implementation generates the covariance matrix and calculates its eigenvectors by numpy routines. Since this implies operations on an dimension*dimension matrix, the execution is especially for the CIFAR-10 data (dimension=3072) very expensive. A possible optimization approach would be to calculate only the n_components first eigenvectors.
The files 'scatterplotMNIST.png' and 'scatterplotCIFAR.png' contain the respective plots for the MNIST and CIFAR-10 datsets.

For MNIST the data looks close the linear seperable. Especially for class 1. For the CIFAR-10 dataset, the first two pca dimension do not suffice to separate the data.

##Problem 22
dA.py contains an implementation of an autoencoder with squared error loss on the MNIST dataset using theano and climin. A autoencoder can be started by the method run_dA(learning_rate=0.1, n_epochs=5, optimizer='gd', n_hidden=500, dataset='mnist.pkl.gz', batch_size=20,n_in = 28 * 28, corruption=0.0).

learning_rate: learning rate, for gradient descent
n_epochs: number of epochs during encoding
optimizer: optimizer, "gd" or "rmsprop"
n_hidden: number of hidden neurons
dataset: filename of the dataset. If the dataset is not present is is downloaded in the folder ../data/. This folder must exist.
batch_size: Batch size
n_in: Number of visible neurons (input and output dimension)
corruption: degree of corruption


Suitable imputs for training the model are 100 hidden units, a batch size of 20, corruption of 0.3 and 5-10 epochs during optimization. More detailled information about training the network is included in the next problem.

##Problem 23
L1 penalty was added to the loss function The method run_dA(..) has now an additional parameter "l1_penalty", which regulates the degree of the penalty (analogous to the hpyerparameter lambda in the problem description).

###Corruption
The following files "filters_rmsprop n_hidden=100corruption=0.0 and l1_pen=0.0.png" and "filters_rmsprop n_hidden=100corruption=0.3 and l1_pen=0.0.png" contain the plots of the respective fields with no corruption (left) and corruption with factor 0.3 (right).

<p align="center">
  <img src="filters_rmsprop n_hidden=100corruption=0.0 and l1_pen=0.0.png"/>
  <img src="filters_rmsprop n_hidden=100corruption=0.3 and l1_pen=0.0.png"/>
</p>

Generally, the respective fields look quite simple. However, in the pictures with corruption the dark parts in the weights are smaller. Single neurons focus on smaller features in the data. 

###L1-penalty
Introducing L1 penalties leads to great results. By forcing most of the hidden units to be close zero, the information about input data is concentrated in only a few nodes. As a consequence the weights of the inner nodes become more and more similar to concrete data.
Following picture contains the respective fields of an autoencoder with 100 hidden units, 0.3 corruption and 0.3 L1 penalty.

<p align="center">
  <img src="filters_rmsprop n_hidden=100corruption=0.3 and l1_pen=0.3.png"/>
</p>

Increasing the l1_penalty factor to 0.6 leads to a even stronger compression. Due to the strong penalty, some nodes do not represent any feature at all but weight that minimize their value best for all datasets. Here even less neurons than given are used to represent the data.

<p align="center">
  <img src="filters_rmsprop n_hidden=100corruption=0.3 and l1_pen=0.6.png"/>
</p>

###Number of hidden nodes
Increasing the number of hidden units yields similar phenomenon than above. With 1600 hidden units and L1 penalties, not all hidden neurons are used to represent the data. The majority of the neurons have similar weights that minimize the value of the neuron. Examples with l1_penalty=0.3 (first) and l1_penalty=0.6 (second) are given here. As a consequence of the stronger penalty, the second run uses less neurons for feature representation.

<p align="center">
  <img src="filters_rmsprop n_hidden=1600corruption=0.3 and l1_pen=0.3.png"/>
  <br/>
  <img src="filters_rmsprop n_hidden=1600corruption=0.3 and l1_pen=0.6.png"/>
</p>

##Problem 24

##Problem 25

the file "autoencoderfilter.png" contains the receptive fields of an execution with 100 inner neurons, rmsprop optimization, corruption of 0.3 and L1_penalty of 0.3. More prints of receptive fields have already been mentioned in Problem 23.

"autoencoderfilter.png":
<p align="center">
  <img src="autoencoderfilter.png"/>
</p>

##Problem 26
Sparse encoding of MNIST means that single pictures are represented by only a few or (in the extreme case) one neurons with strong activations. In this case the inner neurons all corresponde to one class of the MNIST data. This is visible above in the plots of respecitve fields with L1 penalty, which are quite similar to concrete numbers (0-9).