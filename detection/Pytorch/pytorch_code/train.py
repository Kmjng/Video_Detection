# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:56:08 2024

@author: minjeong

tensorboard --logdir=C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/logs

"""
# tensorboard --logdir=C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/logs
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # 신경망 모듈 제공
from torchvision.transforms import transforms
from torch.utils.data import DataLoader # for data loading 
from data_loader import eyes_dataset
from model import Net
import torch.optim as optim # optimizer
import torchvision 

# 모델링 파라미터 저장
#PATH = 'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/pytorch_code/weights/trained.pth'
PATH = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/'


# 데이터 로드
#path1 = r'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/dataset/'
path1 = r"C:/ITWILL/Video_Detection/detection/Pytorch/dataset/"
x_train = np.load(path1 + 'x_train.npy').astype(np.float32)  # (2586, 26, 34, 1)
y_train = np.load(path1 + 'y_train.npy').astype(np.float32)  # (2586, 1)
x_train.shape

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10), # 학습률을 위한 변환
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


'''
# 모델의 예측 결과
from sklearn.metrics import accuracy_score
def get_accuracy(predictions, labels):
    # 예측 값을 0 또는 1로 변환 (이진 분류의 경우)
    predicted_labels = torch.round(torch.sigmoid(predictions)).cpu().numpy()
    true_labels = labels.cpu().numpy()

    # 정확도 계산
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

# 정확도 계산
accuracy = get_accuracy(all_predictions, all_labels)
print(f'Accuracy: {accuracy:.2f}')
'''
#데이터셋을 배치로 나누고, 데이터를 순차적으로 로딩
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
'''
배치 크기 32 
전체 데이터셋 2586개의 샘플 
총 배치 수 약 81(2586 / 32)
'''

model = Net()
model.to('cuda')

criterion = nn.BCEWithLogitsLoss() # 손실함수
optimizer = optim.Adam(model.parameters(), lr=0.0001) # optimizer

epochs = 30


# 텐서보드로 로그 기록
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
writer = SummaryWriter('C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/logs')


for epoch in range(epochs):
    running_loss = 0.0
    running_acc = 0.0

    model.train() # 학습

    for i, data in enumerate(train_dataloader, 0):
        input_1, labels = data[0].to('cuda'), data[1].to('cuda')

        input = input_1.transpose(1, 3).transpose(2, 3)

        optimizer.zero_grad() # epoch 마다 gradient 초기화

        outputs = model(input)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() # weight 업데이트

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)


    #----(추가)-----
    # 에폭이 끝난 후, 평균 손실과 정확도 계산
    avg_loss = running_loss / len(train_dataloader) # 손실/전체 배치 수
    avg_acc = running_acc / len(train_dataloader)
    # 로그 기록
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', avg_acc, epoch)

   
    print('epoch: [%d/%d] train_loss: %.5f train_acc: %.5f' % (
        epoch + 1, epochs, avg_loss, avg_acc))
        

print("learning finish")

torch.save(model.state_dict(), PATH+'cnn_train.pth')

# SummaryWriter 닫기
writer.close()
