# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:55:44 2024

@author: minjeong
"""


import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



##########
# model 1
##########

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.reshape(-1, 1536)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x

##########
# model 2
##########

# 생략

##########
# model 3
##########
'''
from efficientnet_pytorch import EfficientNet

class pre_EffNet(nn.Module):
    def __init__(self, num_classes):
        super(pre_EffNet, self).__init__()
        # EfficientNetB3 불러오기
        self.model = EfficientNet.from_pretrained('efficientnet-b3')
        
        # 첫 번째 합성곱층의 입력 채널 수를 1로 변경
        self.model._conv_stem = nn.Conv2d(
            in_channels=1,  # 1채널 입력
            out_channels=self.model._conv_stem.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

'''
##########
# model 4
##########
import torch

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.quant = torch.quantization.QuantStub()  # 양자화 입력
        self.dequant = torch.quantization.DeQuantStub()  # 양자화 해제 출력
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.fc1 = nn.Linear(128 * 4 * 3, 512)  # Assuming input image size is 32x32 (x) 34x26 (o)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.quant(x)  # Quantize the input
        
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.reshape(-1, 128 * 4 * 3)   # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = self.dequant(x)  # Dequantize the output
        
        return x





#model = Net().to('cuda')
#model = pre_EffNet(num_classes=1).to('cuda') # 마지막 fully connected layer의 출력 채널 수 
model = Net2().to('cuda')
summary(model, (1,26,34))
'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 34]             320
            Conv2d-2           [-1, 64, 13, 17]          18,496
            Conv2d-3            [-1, 128, 6, 8]          73,856
            Linear-4                  [-1, 512]         786,944
            Linear-5                    [-1, 1]             513
================================================================
Total params: 880,129
Trainable params: 880,129
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.37
Params size (MB): 3.36
Estimated Total Size (MB): 3.74
----------------------------------------------------------------
'''

'''
Loaded pretrained weights for efficientnet-b3
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 40, 13, 17]             360
       BatchNorm2d-2           [-1, 40, 13, 17]              80
MemoryEfficientSwish-3           [-1, 40, 13, 17]               0
         ZeroPad2d-4           [-1, 40, 15, 19]               0
Conv2dStaticSamePadding-5           [-1, 40, 13, 17]             360
       BatchNorm2d-6           [-1, 40, 13, 17]              80
MemoryEfficientSwish-7           [-1, 40, 13, 17]               0
          Identity-8             [-1, 40, 1, 1]               0
Conv2dStaticSamePadding-9             [-1, 10, 1, 1]             410
MemoryEfficientSwish-10             [-1, 10, 1, 1]               0
         Identity-11             [-1, 10, 1, 1]               0
Conv2dStaticSamePadding-12             [-1, 40, 1, 1]             440
         Identity-13           [-1, 40, 13, 17]               0
Conv2dStaticSamePadding-14           [-1, 24, 13, 17]             960
      BatchNorm2d-15           [-1, 24, 13, 17]              48
      MBConvBlock-16           [-1, 24, 13, 17]               0
        ZeroPad2d-17           [-1, 24, 15, 19]               0
Conv2dStaticSamePadding-18           [-1, 24, 13, 17]             216
      BatchNorm2d-19           [-1, 24, 13, 17]              48
MemoryEfficientSwish-20           [-1, 24, 13, 17]               0
         Identity-21             [-1, 24, 1, 1]               0
         <..생략..>
         
================================================================
Total params: 10,697,049
Trainable params: 10,697,049
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 9.55
Params size (MB): 40.81
Estimated Total Size (MB): 50.36
----------------------------------------------------------------
'''


'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 34]             320
            Conv2d-2           [-1, 64, 13, 17]          18,496
            Conv2d-3            [-1, 128, 6, 8]          73,856
            Linear-4                  [-1, 512]         786,944
            Linear-5                    [-1, 1]             513
================================================================
Total params: 880,129
Trainable params: 880,129
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.37
Params size (MB): 3.36
Estimated Total Size (MB): 3.74
----------------------------------------------------------------
'''