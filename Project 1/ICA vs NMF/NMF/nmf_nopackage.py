# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:39:51 2019

@author: David
"""
import numpy as np
from PIL import Image
from sys import float_info
import matplotlib.pyplot as plt
import time

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


def setup(X, k=4):
    n, m = X.shape
    W0 = np.random.rand(n, k)
    H0 = np.random.rand(k, m)
    return W0,H0

def runNMF(W0, H0, X, iter_num):

    H = H0
    W = W0
    eps = float_info.min
    V = X

    for i in range(1, iter_num):
        # update H
        # 使用基於歐是距離之目標函數|V-WH|^2之更新規則
        # 利用梯度下降法最佳化
        # 公式如附圖(檔案夾內)
        H = np.multiply(H,(np.dot(W.T,V) / (np.dot(W.T,np.dot(W,H)+ eps))))
        
        # update W
        W = np.multiply(W,(np.dot(V,H.T) / (np.dot(np.dot(W,H),H.T) + eps)))
        
    return H,W
        
samples = datapreprocessing()

#need to flatten
for i in range(len(samples)):
    samples[i] = samples[i].flatten()

samples = np.array(samples)
W0,H0 = setup(samples)

tStart = time.time()#計時開始
H,W = runNMF(W0,H0,samples,1000)
tEnd = time.time()#計時開始結束

print('It cost %f sec' % (tEnd - tStart))

#有沒有再把255乘回來結果都一樣
#H[0] = H[0]*255
plt.imshow(H[0].reshape(28,28))
plt.imshow(H[1].reshape(28,28))
plt.imshow(H[2].reshape(28,28))
plt.imshow(H[3].reshape(28,28))