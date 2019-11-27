# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:47:13 2019

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from sklearn import decomposition
from PIL import Image

#資料前處理
def datapreprocessing():
    #將圖片引入
    imlist = []
    for i in range(1000):
        index = str(r"mixture_dataset\img")+str(i)+str('.jpg')
        imlist.append(index)
      
    #將所有圖normalize
    samples = []
    for i in range(982):
        im = Image.open(imlist[i])
        im_data  = np.empty((im.size[1], im.size[0]))
        for j in range(im.size[1]):
            for k in range(im.size[0]):
                R = im.getpixel((k, j))
                im_data[j,k] = R/255.0
        samples.append(im_data)
    return samples

#for new W
#we update the values of the de-mixing matrix w until the algorithm has converged or the maximum number of iterations has been reached.
#Convergence is considered attained when the dot product of w and its transpose is roughly equal to 1.
def g(x):
    return np.tanh(x)
def g_der(x):
    return 1 - g(x) * g(x)
#update the de-mixing matrix w
def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new

#centering 方便後面處理
def center(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    return X- mean

#whitening: 讓data unrelated
#whitenX = V*X
# V為白化矩陣 ED^-1/2E^T
def Whiten(X):
    #先計算x的covarian matrix
    cov = np.cov(X)
    #找出eigenvector
    d, E = np.linalg.eigh(cov)
    #使之對角化
    #D為特徵值矩陣，為由d組成的對角矩陣
    D = np.diag(d)
    #逆矩陣取根號 
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

def ica(X, iterations, components_nr, tolerance=1e-5):
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)
    for i in range(components_nr):
            print(i)
            # initializes w to some random set
            w = np.random.rand(components_nr)
            for j in range(iterations):
                #iteratively updates w
                w_new = calculate_new_w(w, X)
                #w multiplied by its transpose would be approximately equal to 1
                if i >= 1:
                    w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
                distance = np.abs(np.abs((w * w_new).sum()) - 1)
                w = w_new
                #收斂了就break
                if distance < tolerance:
                    break
            W[i, :] = w
    # 根據求得的W 帶入S=WX, Si為第i時刻的來源訊號
    S = np.dot(W, X)
    return S

samples = datapreprocessing()

#need to flatten
for i in range(len(samples)):
    samples[i] = samples[i].flatten()
    
samples = np.array(samples)

centered = center(samples)
whitened = whiten(centered)

components = 4
S = ica(whitened[:components], 100000, components)

plt.imshow(S[0].reshape(28,28))
plt.imshow(S[1].reshape(28,28))
plt.imshow(S[2].reshape(28,28))
plt.imshow(S[3].reshape(28,28))

#
#plt.imshow(X[0].reshape(28,28))
#plt.imshow(X[1].reshape(28,28))
#plt.imshow(X[2].reshape(28,28))
#plt.imshow(X[3].reshape(28,28))
