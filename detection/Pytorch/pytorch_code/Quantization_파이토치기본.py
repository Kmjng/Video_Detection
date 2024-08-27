# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:08:51 2024

@author: user

파이토치 퓨즈모델 레시피 
https://pytorch.org/tutorials/recipes/fuse.html#define-the-example-model
"""

import torch
import torch.quantization as quant

class AnnotatedConvBnReLUModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvBnReLUModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU(inplace=True)
        # QuantStub and DeQuantStub are not needed for server CPU
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        # x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.dequant(x)
        return x

# Instantiate and prepare model
model = AnnotatedConvBnReLUModel()

# Apply fusion for Conv + BatchNorm + ReLU
model.eval()
torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)

# Apply QConfig for server CPUs
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # or 'qnnpack' if you prefer

# Prepare and calibrate
model = torch.quantization.prepare(model, inplace=False)

# Calibrate your model (using representative data)
def calibrate(model, calibration_data):
    model.eval()
    with torch.no_grad():
        for data in calibration_data:
            model(data)
    return

# Replace [] with your actual calibration data loader or data
calibrate(model, [])

# Convert the model
model = torch.quantization.convert(model, inplace=False)

# Save the TorchScript model
torchscript_model = torch.jit.script(model)
quantized_model_path = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/quantized_model.pth'
torch.jit.save(torchscript_model, quantized_model_path)
