# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:27:40 2024

@author: user

Quantization of EfficientNet-b3 (1) 
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader 
from data_loader import eyes_dataset # ★외부 모듈
from model import pre_EffNet  # Net # , 
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
#PATH = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/cnn_train.pth'
PATH = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/pre_Eff_train.pth'

#path1 = r'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/dataset/'
path1 = r'C:/ITWILL/Video_Detection/detection/Pytorch/dataset/'
x_test = np.load(path1+ 'x_val.npy').astype(np.float32)  # (288, 26, 34, 1)
y_test = np.load(path1+ 'y_val.npy').astype(np.float32)  # (288, 1)

test_transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = eyes_dataset(x_test, y_test, transform=test_transform)

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4) # 배치 8로 수정 

#model = Net()
model = pre_EffNet(1)


# 양자화 준비 (pre_Eff.pth으로부터 )
# 1. 모델을 CPU로 이동
model.to('cpu')
model.load_state_dict(torch.load(PATH)) # train된 모델의 가중치 로드
model.train()  # 양자화 준비를 위해 훈련 모드로 설정


# 손실 함수 정의
criterion = nn.BCEWithLogitsLoss()
count = 0 # 배치 수 count




torch.backends.quantized.engine = 'fbgemm'  # 또는 'qnnpack'
model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm") # backend ='x86', "qnnpack"
model_qat = torch.quantization.prepare_qat(model, inplace=False)

# 샘플 데이터로 양자화 준비
with torch.no_grad():
    for i, test_data in enumerate(test_dataloader, 0):
        data, labels = test_data[0].to('cpu'), test_data[1].to('cpu')
        data = data.transpose(1, 3).transpose(2, 3)
        model_qat(data)  # 샘플 데이터로 양자화 준비
        break

# 양자화
model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)

# 양자화된 모델 저장
quantized_model_path = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/quantized_model.pth'
torch.save(model_qat.state_dict(), quantized_model_path)

print(torch.backends.quantized.engine)  # 양자화 엔진 확인 # x86 backend
print(torch.__version__)  # PyTorch 버전 확인 # 2.0.0

# 텐서보드로 로그 기록
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# ★★★★★★★★★
writer = SummaryWriter('C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/logs/quantized/pre_Eff/test')


# 학습 전 GPU 메모리 사용량 측정
start_memory = torch.cuda.memory_allocated()


model_qat.eval() 

with torch.no_grad():
    total_loss = 0.0
    total_acc = 0.0
    acc = 0.0
    for i, test_data in enumerate(test_dataloader, 0):
        data, labels = test_data[0].to('cpu'), test_data[1].to('cpu')

        data = data.transpose(1, 3).transpose(2, 3)

        outputs = model_qat(data)

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
        
# 학습 후 GPU 메모리 사용량 측정
end_memory = torch.cuda.memory_allocated()

print("Validation finish")
# 전체 학습에 사용된 메모리 출력
print(f'(Quantized) Total memory used during validation: {end_memory - start_memory} bytes')

# SummaryWriter 닫기
writer.close()