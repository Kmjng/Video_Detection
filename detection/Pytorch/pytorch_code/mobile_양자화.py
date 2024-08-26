# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:03:15 2024

@author: user
"""

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

class AnnotatedConvBnReLUModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvBnReLUModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU(inplace=True)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

model = AnnotatedConvBnReLUModel()

torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)

model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)
# Calibrate your model
def calibrate(model, calibration_data):
    # Your calibration code here
    return
calibrate(model, [])
torch.quantization.convert(model, inplace=True)


torchscript_model = torch.jit.script(model)


torchscript_model_optimized = optimize_for_mobile(torchscript_model)
torch.jit.save(torchscript_model_optimized, "model.pt")
