# -*- coding: utf-8 -*-
"""
@author: David
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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

def cal_pairwise_dist(x):
    '''計算pairwise距離、x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #return 任意兩點間距離平方
    return dist

#尋找每個樣本點的n個近鄰點；
def get_n_neighbors(data, n_neighbors):
    '''
    :param data: (n_samples, n_features)
    :param n_neighbors: n nearest neighbors
    :return: neighbors indexs
    '''
    dist = cal_pairwise_dist(data)**0.5
    n = dist.shape[0]
    N = np.zeros((n, n_neighbors))

    for i in range(n):
        index_ = np.argsort(dist[i])[1:n_neighbors+1]
        N[i] = N[i] + index_

    return N.astype(np.int32)

def lle(data, n_dims, n_neighbors):
    '''
    :param data:(n_samples, n_features)
    :param n_dims: target n_dims
    :param n_neighbors: n nearest neighbors
    :return: (n_samples, n_dims)
    '''
    N = get_n_neighbors(data, n_neighbors)
    n, D = data.shape

    # prevent Si to small
    if n_neighbors > D:
        tol = 1e-3
    else:
        tol = 0
    # calculate W
    # 由每個樣本點的近鄰點計算出該樣本點的區域性重建權值矩陣
    W = np.zeros((n_neighbors, n))
    I = np.ones((n_neighbors, 1))
    for i in range(n):
        Xi = np.tile(data[i], (n_neighbors, 1)).T
        Ni = data[N[i]].T

        Si = np.dot((Xi-Ni).T, (Xi-Ni))
        Si = Si+np.eye(n_neighbors)*tol*np.trace(Si)

        Si_inv = np.linalg.pinv(Si)
        wi = (np.dot(Si_inv, I))/(np.dot(np.dot(I.T, Si_inv), I)[0,0])
        W[:, i] = wi[:,0]
#    print("Xi.shape", Xi.shape)
#    print("Ni.shape", Ni.shape)
#    print("Si.shape", Si.shape)
    # 由該樣本點的區域性重建權值矩陣和其近鄰點計算出該樣本點的輸出值Y
    W_y = np.zeros((n, n))
    for i in range(n):
        index = N[i]
        for j in range(n_neighbors):
            W_y[index[j],i] = W[j,i]

    I_y = np.eye(n)
    M = np.dot((I_y - W_y), (I_y - W_y).T)

    eig_val, eig_vector = np.linalg.eig(M)
    index_ = np.argsort(np.abs(eig_val))[1:n_dims+1]
#    print("index_", index_)
    Y = eig_vector[:, index_]
    return Y

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

afterlle =lle(flatten, 2, 6)

show(afterlle, flatten)