import numpy as np
from numpy.linalg import norm
from time import time
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

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


samples = datapreprocessing()

#need to flatten
for i in range(len(samples)):
    samples[i] = samples[i].flatten()
    
model = NMF(n_components=4, init='random', random_state=0)
W = model.fit_transform(samples)
H = model.components_

plt.imshow(H[0].reshape(28,28))
plt.imshow(H[1].reshape(28,28))
plt.imshow(H[2].reshape(28,28))
plt.imshow(H[3].reshape(28,28))
#plt.imshow(H[4].reshape(28,28))