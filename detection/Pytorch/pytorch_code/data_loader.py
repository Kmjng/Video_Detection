# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:42:26 2024

@author: minjeong

이 eyes_dataset 클래스는 PyTorch의 Dataset 클래스를 상속하여, 
이미지 파일 경로와 라벨 파일 경로를 관리하는 데이터셋을 정의
"""
from torch.utils.data import Dataset
import torch

# data_loader 



class eyes_dataset(Dataset): # 사용자 정의 데이터셋 만들기 
    def __init__(self, x_file_paths, y_file_path, transform=None):
        self.x_files = x_file_paths
        self.y_files = y_file_path
        self.transform = transform # 이미지 데이터 x_files 를 Pytorch 텐서로 변환 

    def __getitem__(self, idx):
        x = self.x_files[idx] #  idx 위치에 있는 데이터 항목을 반환
        x = torch.from_numpy(x).float()  # 데이터 항목 텐서화 
        y = self.y_files[idx] #  idx 위치에 있는 레이블을 반환
        y = torch.from_numpy(y).float() # 레이블 항목 텐서화 

        return x, y

    def __len__(self):
        return len(self.x_files)

