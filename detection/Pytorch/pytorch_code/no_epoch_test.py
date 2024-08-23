# -*- coding: utf-8 -*-
"""
에폭 없는 validation 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader 
from data_loader import eyes_dataset # ★외부 모듈
from model import Net # , pre_EffNet  
import torch.optim as optim

    
    
# accuracy 함수
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


#PATH = 'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/pytorch_code/weights/trained.pth'
# ★★★★★★★★★
PATH = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/cnn_train.pth'

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
#model = pre_EffNet(1)

model.to('cuda')
model.load_state_dict(torch.load(PATH)) # train된 모델의 가중치 로드
model.eval() # 평가 모드 

count = 0 # 배치 수 count

# 손실 함수 정의
criterion = nn.BCEWithLogitsLoss()

# 텐서보드로 로그 기록
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# ★★★★★★★★★
writer = SummaryWriter('C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/logs/cnn_model_log/test')



with torch.no_grad():
    total_loss = 0.0
    total_acc = 0.0
    acc = 0.0
    for i, test_data in enumerate(test_dataloader, 0):
        data, labels = test_data[0].to('cuda'), test_data[1].to('cuda')

        data = data.transpose(1, 3).transpose(2, 3)

        outputs = model(data)

        acc = accuracy(outputs, labels)
        total_acc += acc
        
        # 손실 계산
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        count = i
    
    # 평균 손실과 정확도 계산
    avg_loss = total_loss / count
    avg_acc = total_acc / count
    # 텐서보드에 기록
    writer.add_scalar('Loss/validation', avg_loss)
    writer.add_scalar('Accuracy/validation', avg_acc)

    print(f'Validation Loss: {avg_loss:.5f}')
    print(f'Validation Accuracy: {avg_acc:.5f}%')
    print('average acc: %.5f' % (total_acc/count),'%')
    
print('Validation finished!')

# SummaryWriter 닫기
writer.close()