# -*- coding: utf-8 -*-
"""
@author: David
"""
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from numpy.linalg import eig
    
def datapreprocessing():
    data = np.load('digits-labels.npz')
    image = data['d']
    label = data['l']
    fivedigit = []
    fiveimage = []
    for i in range(len(label)):
        if label[i]==5:
            fivedigit.append(image[:,i])
    
    fivedigit = np.array(fivedigit)
    for i in range(len(fivedigit)):
        fiveimage.append(fivedigit[i].reshape(28,28))
        
    fiveimage = np.array(fiveimage)
    return fivedigit, fiveimage

def PCA(data):
    # calculate the mean of each column
    meaned = np.mean(data.T, axis=1)
    # center columns by subtracting column means
    centered = data - meaned
    # calculate covariance matrix of centered matrix
    V = np.cov(centered.T)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    # project data
    P = vectors.T.dot(centered.T)
    return P.T


def show(transform, data):
    '''
        size of transform : (n, 2)
        size of data : (n, 784)
        '''
    fig, ax = plt.subplots()
    ax.scatter(transform[:, 0], transform[:, 1])
    for x0, y0, img in zip(transform[:, 0], transform[:, 1], data.reshape((-1, 28, 28), order='F')):
        ab = AnnotationBbox(OffsetImage(img, zoom=0.4, cmap='gray'), (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.show()
    
flatten, samples = datapreprocessing()

aftpca = PCA(flatten).real[:,:2]

show(aftpca,flatten)