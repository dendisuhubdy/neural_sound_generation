import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import math

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def show_histogram(X, y, label_dict, feature_dict):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for cnt in range(4):
            plt.subplot(2, 2, cnt+1)
            for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
                plt.hist(X[y==lab, cnt],
                         label=lab,
                         bins=10,
                         alpha=0.3,)
            plt.xlabel(feature_dict[cnt])
        plt.legend(loc='upper right', fancybox=True, fontsize=8)
        plt.tight_layout()
        plt.savefig('./histogram_iris.png')
        #plt.show()
    
def pca_np(X, y, label_dict, feature_dict):
    # standarize data with mean 0 and variance 1
    X_std = StandardScaler().fit_transform(X)

    # calculate the mean vector
    mean_vec = np.mean(X_std, axis=0)
    # compute the covariance matrix
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    print('Covariance matrix \n%s' %cov_mat)

    # or using Numpy covariance matrix
    # print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
    # perform eigendecomposition on the covariance matrix
    # cov_mat = np.cov(X_std.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)

    # Eigendecomposition of the standardized data
    # based on the correlation matrix

    # singular vector decomposition
    u,s,v = np.linalg.svd(X_std.T)
    for ev in eig_vecs.T:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('Everything ok!')
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])

    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))

        plt.bar(range(4), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(4), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig("./explained_variance.png")

    matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                          eig_pairs[1][1].reshape(4,1)))

    print('Matrix W:\n', matrix_w)

    # project this to a new space

    Y = X_std.dot(matrix_w)

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                            ('blue', 'red', 'green')):
            plt.scatter(Y[y==lab, 0],
                        Y[y==lab, 1],
                        label=lab,
                        c=col)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='lower center')
        plt.tight_layout()
        plt.savefig("./pca_result_numpy.png")

    print(Y)
    return Y


def pca_sklearn(X, y, label_dict, feature_dict):
    X_std = StandardScaler().fit_transform(X)
    sklearn_pca = sklearnPCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                            ('blue', 'red', 'green')):
            plt.scatter(Y_sklearn[y==lab, 0],
                        Y_sklearn[y==lab, 1],
                        label=lab,
                        c=col)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='lower center')
        plt.tight_layout()
        plt.savefig("./pca_result_sklearn.png")
    
    print(Y_sklearn)
    return Y_sklearn

if __name__ == "__main__":
    df = pd.read_csv(
	filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
	header=None,
	sep=',')

    df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    df.dropna(how="all", inplace=True) # drops the empty line at file-end

    df.tail()

    # split data table into data X and class labels y

    X = df.ix[:,0:4].values
    y = df.ix[:,4].values

    label_dict = {1: 'Iris-Setosa',
		  2: 'Iris-Versicolor',
		  3: 'Iris-Virgnica'}

    feature_dict = {0: 'sepal length [cm]',
		    1: 'sepal width [cm]',
		    2: 'petal length [cm]',
		    3: 'petal width [cm]'}

    show_histogram(X, y, label_dict, feature_dict)

    result1 = pca_np(X, y, label_dict, feature_dict)
    result2 = pca_sklearn(X, y, label_dict, feature_dict)
