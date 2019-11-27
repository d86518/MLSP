# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:04:54 2019

@author: David
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from PIL import Image

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

class FastICA2(FastICA):    
    def mixing(self):
        return self.mixing_.T


samples = datapreprocessing()
for i in range(len(samples)):
    samples[i] = samples[i].flatten()
ica = FastICA2(n_components=4,max_iter=10000)
Snew = ica.fit_transform(samples)

mix = ica.mixing()

plt.imshow(mix[0].reshape(28,28))
plt.imshow(mix[1].reshape(28,28))
plt.imshow(mix[2].reshape(28,28))
plt.imshow(mix[3].reshape(28,28))

