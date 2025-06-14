import cv2
import numpy as np
import torch.nn as nn

from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch
transform1 = transforms.ToPILImage()
transform2 = transforms.ToTensor()

def adjust_contrast(image_tensor, factor):
    """
    调节图像对比度
    :param image_path: 输入图像路径
    :param factor: 对比度因子（>1 增强对比度，<1 降低对比度）
    :return: 调节后的图像
    """
    image_sum = []
    for i in image_tensor:
        image = transform1(i)
        enhancer = ImageEnhance.Contrast(image)
        adjusted_image = enhancer.enhance(factor)
        adjusted_image = transform2(adjusted_image)
        image_sum.append(adjusted_image)
    image_ = torch.stack(image_sum, dim = 0)
    return image_.to('cuda')

class Contrast_adapter(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(n_dim, n_dim, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(n_dim, n_dim, kernel_size=1, padding=0, stride=1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x