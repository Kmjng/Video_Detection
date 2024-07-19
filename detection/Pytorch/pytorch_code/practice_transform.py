import numpy as np
from PIL import Image
from torchvision import transforms

# 원본 데이터
data = [[1, 2], [3, 4]]

# NumPy 배열로 변환
data_np = np.array(data, dtype=np.float32)

# NumPy 배열을 PIL 이미지로 변환 (1채널 그레이스케일 이미지로 가정)
data_img = Image.fromarray(data_np)

# 'data_img'를 (1, 2) 형태의 PIL 이미지로 변환
data_img = data_img.convert('L')


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
])

# [과정1] 변환 적용
transformed_img = train_transform(data_img)

# [과정2] 변환된 결과를 NumPy 배열로 변환 (옵션)
transformed_np = transformed_img.numpy()

# 결과 출력
print("Original Data:\n", data_np)
print("Transformed Data (Tensor):\n", transformed_img)
print("Transformed Data (NumPy):\n", transformed_np)
'''
Original Data:
 [[1. 2.]
 [3. 4.]]
Transformed Data (Tensor):
 tensor([[[0.0078, 0.0039],
         [0.0157, 0.0118]]])
Transformed Data (NumPy):
 [[[0.00784314 0.00392157]
  [0.01568628 0.01176471]]]
'''

# 이미지 출력
import matplotlib.pyplot as plt 
plt.figure(figsize=(10, 5))

# 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(data_img, cmap='gray')
plt.axis('off')

# 변환된 이미지 출력
plt.subplot(1, 2, 2)
plt.title("Transformed Image")
plt.imshow(transformed_np.squeeze(), cmap='gray')
plt.axis('off')

plt.show()
