# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:21:58 2024

@author: minjeong

dataset.csv => .npy
"""

from imagetags import *
import matplotlib.pyplot as plt
import os, glob, cv2, random
import seaborn as sns
import pandas as pd

base_path = 'C:/ITWILL/Video_Detection/detection/Pytorch/dataset/'

X, y = read_csv(os.path.join(base_path, 'dataset.csv'))

print(X.shape, y.shape)

# 내가 봤을때 왼쪽 눈
plt.figure(figsize=(12, 10))
for i in range(50):
    plt.subplot(10, 5, i+1)
    plt.axis('off')
    plt.imshow(X[i].reshape((26, 34)), cmap='gray')
    
sns.distplot(y, kde=False)


# 전처리 
n_total = len(X)
X_result = np.empty((n_total, 26, 34, 1)) 

for i, x in enumerate(X):
    img = x.reshape((26, 34, 1)) # # gray scaling
    
    X_result[i] = img
    
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_result, y, test_size=0.1)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

np.save(base_path + 'x_train.npy', x_train)
np.save(base_path + 'y_train.npy', y_train)
np.save(base_path + 'x_val.npy', x_val)
np.save(base_path + 'y_val.npy', y_val)






    