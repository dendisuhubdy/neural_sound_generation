import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

"""
def pca_np():
    x = np.array([
	    [0.387,4878, 5.42],
	    [0.723,12104,5.25],
	    [1,12756,5.52],
	    [1.524,6787,3.94],
	])

    #centering the data
    x -= np.mean(x, axis = 0)  

    cov = np.cov(x, rowvar = False)

    evals , evecs = LA.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    a = np.dot(x, evecs) 
    print(a)
"""

def apply_pca(x):
    pca = decomposition.PCA(n_components=3)

    # x = np.array([
            # [0.387,4878, 5.42],
            # [0.723,12104,5.25],
            # [1,12756,5.52],
            # [1.524,6787,3.94],
        # ])
    
    # https://stats.stackexchange.com/questions/235882/pca-in-numpy-and-sklearn-produces-different-results
    # pca.fit_transform(x)

    x_std = StandardScaler().fit_transform(x)
    result = pca.fit_transform(x_std)
    return result
