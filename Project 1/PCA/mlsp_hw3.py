# -*- coding: utf-8 -*-
"""
MLSP HW3 PCA 2D_2D
@author: David
"""

import numpy as np
from PIL import Image

#資料前處理
def datapreprocessing():
    #將圖片引入
    imlist = []
    for i in range(982):
        index = str(r"four_dataset\four")+str(i)+str('.jpg')
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

def PCA(row_top, col_top):
    #先進行前處理
    samples = datapreprocessing()
    print(np.array(samples).shape)
    size = samples[0].shape
    # m*n matrix
    
    # 將所有samples取mean 讓後面做mean normalization運算會比較方便
    mean = np.zeros(size)

    for s in samples:
        mean = mean + s

    mean /= float(len(samples))

    # 求 covariance matrix 
    # n*n matrix 
    cov_row = np.zeros((size[1],size[1]))
    for s in samples:
        diff = s - mean;
        cov_row += np.dot(diff.T, diff)
    cov_row /= float(len(samples))
    
    #eval for eigenvalue
    #evec for eigenvector
    row_eval, row_evec = np.linalg.eig(cov_row)
    
    # 選eig values值最高者，先進行sort
    sorted_index = np.argsort(row_eval)
    # using slice operation to reverse
    X = row_evec[:,sorted_index[:-row_top-1 : -1]]

    # m*m matrix
    cov_col = np.zeros((size[0], size[0]))
    for s in samples:
        diff = s - mean;
        cov_col += np.dot(diff,diff.T)
    cov_col /= float(len(samples))
    
    #eval for eigenvalue
    #evec for eigenvector    
    col_eval, col_evec = np.linalg.eig(cov_col)
    sorted_index = np.argsort(col_eval)
    Z = col_evec[:,sorted_index[:-col_top-1 : -1]]

    return X, Z, samples


#X為行方向上2D PCA得到的變換矩陣，Z為列方向上2D PCA得到的變換矩陣
#num為第幾張圖，row/col為壓縮的dimension
def transform(num,row,col):
    X, Z, samples = PCA(row, col)
    
    #選擇four0(samples[0])
    res = np.dot(Z.T, np.dot(samples[num], X))
    res = np.dot(Z, np.dot(res, X.T))
#    print(res,res.shape)
    
    #mode L 為 new 一個 image for 8-bit pixels, black and white
    row_im = Image.new('L', (res.shape[1], res.shape[0]))
    y=res.reshape(1, res.shape[0]*res.shape[1])
    
    #將資料反normalize save起來
    row_im.putdata([int(t*255) for t in y[0].tolist()])
    row_im.save(str(row)+'x'+str(col)+'.png')
    
transform(0,16,16)