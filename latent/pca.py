import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.utils import shuffle

from load_data import load_data

def doPCA(data, n_components=2):

    # Subtract mean and build covariance matrix
    data2 = data - np.mean(data, axis=0)
    covarance = np.cov(data2.transpose())

    #Calculate eigenvalues and -vectors
    W, V = np.linalg.eig(covarance)

    #Sort eigenvalues and respective eigenvectors
    zipped = zip(W, V.transpose())
    rearrangedEvalsVecs = sorted(zipped,key=lambda x: x[0].real, reverse=True)
    W,V = zip(*rearrangedEvalsVecs)
    print('... pca performed')

    #Return first n_components components
    return np.asarray(V).transpose()[:,:n_components]

def doPCAScatterplot(dataset='mnist.pkl.gz', n_samples=30000):
    datasets = load_data(dataset)

    X_train, y_train = datasets[0]

    X_train, y_train = shuffle(X_train, y_train)
    X_train, y_train = X_train[:n_samples], y_train[:n_samples]

    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    #plt.prism()

    print('...training started.')

    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue


        X_ = X_train[(y_train == i) + (y_train == j)]
        y_ = y_train[(y_train == i) + (y_train == j)]

        PCA= doPCA(X_)
        X_transformed = np.dot(X_, PCA)
        plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_, marker='o',edgecolor='none')
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())

        plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_, marker='o',edgecolor='none')
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            plots[i, j].set_title(j)
            plots[j, i].set_ylabel(j)

    plt.tight_layout()

    plt.savefig("scatterplotCIFAR.png")
    #plt.savefig("scatterplotNMIST.png")

if __name__ == "__main__":
    doPCAScatterplot('cifar-10-python.tar.gz')
    #doPCAScatterplot('mnist.pkl.gz')