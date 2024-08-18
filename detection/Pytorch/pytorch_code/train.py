# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:56:08 2024

@author: minjeong
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import eyes_dataset
from model import Net
import torch.optim as optim

path1 = r'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/dataset/'
x_train = np.load(path1 + 'x_train.npy').astype(np.float32)  # (2586, 26, 34, 1)
y_train = np.load(path1 + 'y_train.npy').astype(np.float32)  # (2586, 1)
x_train.shape
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(), #모델 일반화 능력 향상시키기 위함

])

train_dataset = eyes_dataset(x_train, y_train, transform=train_transform)

# --------데이터 출력----------
'''
plt.style.use('dark_background')
fig = plt.figure()

for i in range(len(train_dataset)):
    x, y = train_dataset[i]

    plt.subplot(2, 1, 1)
    plt.title(str(y_train[i]))
    plt.imshow(x_train[i].reshape((26, 34)), cmap='gray')

    plt.show()
    '''
    
    
# accuracy 함수
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


    
PATH = 'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/pytorch_code/weights/trained.pth'

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

model = Net()
model.to('cpu')

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 50

for epoch in range(epochs):
    running_loss = 0.0
    running_acc = 0.0

    model.train()

    for i, data in enumerate(train_dataloader, 0):
        input_1, labels = data[0].to('cpu'), data[1].to('cpu')

        input = input_1.transpose(1, 3).transpose(2, 3)

        optimizer.zero_grad()

        outputs = model(input)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)

        if i % 80 == 79:
            print('epoch: [%d/%d] train_loss: %.5f train_acc: %.5f' % (
                epoch + 1, epochs, running_loss / 80, running_acc / 80))
            running_loss = 0.0

print("learning finish")
torch.save(model.state_dict(), PATH)