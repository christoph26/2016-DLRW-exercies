##Problem 20

The file "pca.py" contains an implementation of pca. The actual pca is done in the method doPCA(data, n_components=2). Data is an array with the for the PCA, n_components is the number of eigenvectors, which should be returned. the returned array can be used to truncate orginal data.

##Problem 21
The method doPCAScatterplot(dataset='mnist.pkl.gz', n_samples=30000) performs the scatterplot. dataset specifies with 'mnist.pkl.gz' or 'cifar-10-python.tar.gz' the dataset, n_samples the number of randomly select samples for the plot. The current implementation generates the covariance matrix and calculates its eigenvectors by numpy routines. Since this implies operations on an dimension*dimension matrix, the execution is especially for the CIFAR-10 data (dimension=3072) very expensive. A possible optimization approach would be to calculate only the n_components first eigenvectors.
The files 'scatterplotMNIST.png' and 'scatterplotCIFAR.png' contain the respective plots for the MNIST and CIFAR-10 datsets.