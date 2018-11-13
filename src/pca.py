import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy import linalg as LA


def run_pca_np(x):
    #centering the data
    x -= np.mean(x, axis = 0)  

    cov = np.cov(x, rowvar = False)

    evals , evecs = LA.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    a = np.dot(x, evecs) 
    return a

def run_pca(x, n_components=3):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    print("Before normalization")
    mean = np.mean(x)
    variance = np.var(x)
    print(mean)
    print(variance)
    scaler.fit(x)
    # print(scaler.mean_)
    x_std = scaler.transform(x)
    print("After normalization")
    mean_std = np.mean(x_std)
    variance_std = np.var(x_std)
    print(mean_std)
    print(variance_std)
    # print(x_std)
    pca = decomposition.PCA(n_components)
    # https://stats.stackexchange.com/questions/235882/pca-in-numpy-and-sklearn-produces-different-results
    # input the standarized data with mean 0 and var 1 to the PCA
    result = pca.fit_transform(x_std)
    return result


if __name__ == "__main__":
    x = np.random.random_sample((18, 3209))
    print(x)
    print(x.shape)
    result = run_pca(x)
    print(result)
    print(result.shape)
