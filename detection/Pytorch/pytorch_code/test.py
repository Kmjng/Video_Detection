# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:58:10 2024

@author: minjeong
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader 
from data_loader import eyes_dataset # ★외부 모듈
from model import Net
import torch.optim as optim

    
    
# accuracy 함수
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


#PATH = 'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/pytorch_code/weights/trained.pth'
PATH = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/trained.pth'

#path1 = r'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/dataset/'
path1 = r'C:/ITWILL/Video_Detection/detection/Pytorch/dataset/'
x_test = np.load(path1+ 'x_val.npy').astype(np.float32)  # (288, 26, 34, 1)
y_test = np.load(path1+ 'y_val.npy').astype(np.float32)  # (288, 1)

test_transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = eyes_dataset(x_test, y_test, transform=test_transform)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

model = Net()
model.to('cuda')
model.load_state_dict(torch.load(PATH)) # train된 모델의 가중치 로드
model.eval() # 평가 모드 

count = 0 # 배치 수 count

with torch.no_grad():
    total_acc = 0.0
    acc = 0.0
    for i, test_data in enumerate(test_dataloader, 0):
        data, labels = test_data[0].to('cuda'), test_data[1].to('cuda')

        data = data.transpose(1, 3).transpose(2, 3)

        outputs = model(data)

        acc = accuracy(outputs, labels)
        total_acc += acc

        count = i

    print('avarage acc: %.5f' % (total_acc/count),'%')

print('test finish!')