# -*- coding: utf-8 -*-
"""
@author: David
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd

    
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


def floyd(D,n_neighbors):
    '''
    用floyd計算最短距離
    '''
    Max = np.max(D)*1000
    n1,n2 = D.shape
    k = n_neighbors
    D1 = np.ones((n1,n1))*Max
    D_arg = np.argsort(D,axis=1)
    for i in range(n1):
        D1[i,D_arg[i,0:k+1]] = D[i,D_arg[i,0:k+1]]
    for k in range(n1):
        for i in range(n1):
            for j in range(n1):
                if D1[i,k]+D1[k,j]<D1[i,j]:
                    D1[i,j] = D1[i,k]+D1[k,j]
    return D1

def cal_pairwise_dist(x):
    '''計算pairwise距離、x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #return 任意兩點間距離平方
    return dist

def my_mds(dist, n_dims):
    '''
    通過MDS構建低維的資料嵌入
    '''
    # dist (n_samples, n_samples)
    dist = dist**2
    n = dist.shape[0]
    # global T1, T2 , T3
    #分別為disti,distj,distij項
    T1 = np.ones((n,n))*np.sum(dist)/n**2
    T2 = np.sum(dist, axis = 1)/n
    T3 = np.sum(dist, axis = 0)/n

    # 按公式計算，令B為距離矩陣Z^T*Z
    B = -(T1 - T2 - T3 + dist)/2

    # 對距離矩陣B做特徵值分解，則可以獲得Z表示式
    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]

    return picked_eig_vector*picked_eig_val**(0.5)


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

D = cal_pairwise_dist(flatten)**0.5
#有nan情形可能是數字過小，先以0代替
D = pd.DataFrame(D).fillna(0)
D = np.array(D)
#floyd由於資料量大會跑得滿久的
D_floyd=floyd(D, n_neighbors = 6 )
data_n = my_mds(D_floyd, n_dims = 2)
#取real來計算，也可以直接帶入
afterisomap = data_n.real

show(afterisomap, flatten)