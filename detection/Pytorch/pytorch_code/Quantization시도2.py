# -*- coding: utf-8 -*-
"""
<<<test>>>

Quantization of cnn
for test
정적 양자화
 이미 학습된 모델을 양자화
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader 
from data_loader import eyes_dataset # ★외부 모듈
from model import Net2 # ,   pre_EffNet  # 
import torch.optim as optim
import torch.quantization as quant
    
    
# accuracy 함수
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc




#path1 = r'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/dataset/'
path1 = r'C:/ITWILL/Video_Detection/detection/Pytorch/dataset/'
x_test = np.load(path1+ 'x_val.npy').astype(np.float32)  # (288, 26, 34, 1)
y_test = np.load(path1+ 'y_val.npy').astype(np.float32)  # (288, 1)

# Adjust the dimensions of x_test
x_test = np.transpose(x_test, (0, 3, 1, 2))  # Change from (N, H, W, C) to (N, C, H, W)
# input 텐서의 차원 순서를 (배치 크기, 채널, 높이, 너비)로

test_transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = eyes_dataset(x_test, y_test, transform=test_transform)

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4) # 배치 8로 수정 

model = Net2()
#model = pre_EffNet(1)  

# 손실 함수 정의
criterion = nn.BCEWithLogitsLoss()
count = 0 # 배치 수 count

# 양자화 준비 (pre_Eff.pth으로부터 )
# 1. 모델을 CPU로 이동

# cnn_train, pre_Eff_train
state_dict = torch.load('C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/cnn2_train.pth')
model.load_state_dict(state_dict, strict=False)  # strict=False를 사용하여 일부 키가 누락된 경우 무시
model.to('cpu')  # 모델을 CPU로 이동
model.eval()  # 평가 모드로 설정

# Apply fusion for Conv + BatchNorm + ReLU
# Conv2d + BatchNorm2d + ReLU
torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu1']], inplace=True)
torch.quantization.fuse_modules(model, [['conv2', 'bn2', 'relu2']], inplace=True)
torch.quantization.fuse_modules(model, [['conv3', 'bn3', 'relu3']], inplace=True)

# Apply QConfig for server CPUs
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # or 'qnnpack' if you prefer

# Prepare and calibrate
model = torch.quantization.prepare(model, inplace=False)

# Calibration
def calibrate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to('cpu')  # Ensure inputs are on CPU
            model(inputs)

# Calibrate with test data (or representative data)
calibrate(model, test_dataloader)

# Convert the model
model = torch.quantization.convert(model, inplace=False)


# TorchScript 모델로 저장
torchscript_model = torch.jit.script(model)
quantized_model_path = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/quantized_model.pth'
torch.jit.save(torchscript_model, quantized_model_path)


print(torch.backends.quantized.engine)  # 양자화 엔진 확인 # 초기값: x86 backend
print(torch.__version__)  # PyTorch 버전 확인 # 2.0.0



# -----------------------------------------------------
loaded_quantized_model = torch.jit.load(quantized_model_path)
# Evaluate the quantized model
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to('cpu')  # Ensure inputs are on CPU
            labels = labels.to('cpu')  # Ensure labels are on CPU
            outputs = model(inputs)
            all_preds.append(outputs)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute accuracy
    acc = accuracy(all_preds, all_labels)
    return acc


# 현재 프로세스의 메모리 사용량 측정
import psutil
import os

process = psutil.Process(os.getpid())
# 학습 전 CPU 메모리 사용량 측정
start_memory = process.memory_info().rss  # 메모리 사용량 (바이트 단위)


# Compute accuracy on the test set

test_accuracy = evaluate(loaded_quantized_model, test_dataloader)

# 학습 후 메모리 사용량 측정
end_memory = process.memory_info().rss  # 메모리 사용량 (바이트 단위)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print("--Validation finish--")
# 전체 학습에 사용된 메모리 출력
print(f'(Quantized) Total memory used during validation: {end_memory - start_memory} bytes')


'''
# 텐서보드로 로그 기록
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# ★★★★★★★★★
writer = SummaryWriter('C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/logs/quantized/cnn2/test')


# 현재 프로세스의 메모리 사용량 측정
import psutil
import os

process = psutil.Process(os.getpid())
# 학습 전 CPU 메모리 사용량 측정
start_memory = process.memory_info().rss  # 메모리 사용량 (바이트 단위)


model.eval() 

with torch.no_grad():
    total_loss = 0.0
    total_acc = 0.0
    acc = 0.0
    for i, test_data in enumerate(test_dataloader, 0):
        data, labels = test_data[0].to('cpu'), test_data[1].to('cpu')

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
        
# 학습 후 메모리 사용량 측정
end_memory = process.memory_info().rss  # 메모리 사용량 (바이트 단위)

print("Validation finish")
# 전체 학습에 사용된 메모리 출력
print(f'(Quantized) Total memory used during validation: {end_memory - start_memory} bytes')

# SummaryWriter 닫기
writer.close()
'''