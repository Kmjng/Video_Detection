# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:48:18 2024

@author: user
writer = SummaryWriter('C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/logs')
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os 

# TensorBoard 로그 파일 경로
log_dir = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/logs'
#event_file = None
event_file = os.path.join(log_dir, 'events.out.tfevents.1724212626.DESKTOP-737DR38.49324.1')

# 로그 파일에서 가장 최근의 파일을 선택 (여기서는 하나만 가정)
'''
for file in os.listdir(log_dir):
    if file.startswith('events.out.tfevents'):
        event_file = os.path.join(log_dir, file)

if event_file is None:
    raise ValueError("No TensorBoard event files found")
'''
# TensorBoard 로그 파일 읽기
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# 'Accuracy/train' 스칼라 데이터 읽기
tags = ea.Tags()['scalars']
if 'Accuracy/train' not in tags:
    raise ValueError("'Accuracy/train' tag not found in the logs")

# 스칼라 값 가져오기
accuracy_events = ea.Scalars('Accuracy/train')
steps = [e.step for e in accuracy_events]
values = [e.value for e in accuracy_events]

# 데이터 플로팅
plt.figure(figsize=(10, 6))
plt.plot(steps, values, label='Train Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)

# 이미지 파일로 저장
plt.savefig('train_accuracy.png')

# 그래프 보여주기 (선택 사항)
plt.show()
