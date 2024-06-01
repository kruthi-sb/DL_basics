import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def load_planar_dataset(m=400):
    np.random.seed(1)
    #m = 400 # number of examples
    N = int(m/2) # number of points per class
    nx = 2 # dimensionality
    X = np.zeros((m,nx)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets(m=200):  
    noisy_circles = sklearn.datasets.make_circles(n_samples=m, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=m, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=m, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=m, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(m, 2), np.random.rand(m, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure