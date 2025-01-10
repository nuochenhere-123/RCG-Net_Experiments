import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import cv2
import os
import numpy as np
from Config.options import opt
import torchvision
import torchvision.transforms.functional as F
import numbers
import random
from PIL import Image



class ToTensor(object):
    def __call__(self, sample):
        hazy_image, clean_image = sample['raw'], sample['reference']
        hazy_image = torch.from_numpy(np.array(hazy_image).astype(np.float32))
        hazy_image = torch.transpose(torch.transpose(hazy_image, 2, 0), 1, 2)
        # hazy_image = hazy_image / 255.0
        clean_image = torch.from_numpy(np.array(clean_image).astype(np.float32))
        clean_image = torch.transpose(torch.transpose(clean_image, 2, 0), 1, 2)
        # clean_image = clean_image / 255.0
        return {'raw': hazy_image,
                'reference': clean_image}


class Dataset_Load(Dataset):
    def __init__(self, hazy_path, clean_path, transform=None):
        # 获取 hazy 和 clean 文件夹中的所有图片文件名
        self.hazy_dir = hazy_path
        self.clean_dir = clean_path
        self.transform = transform
        
        # 列出所有图片文件（支持 .png 和 .jpg）
        self.hazy_images = sorted([f for f in os.listdir(self.hazy_dir) if f.endswith(('png', 'jpg'))])
        self.clean_images = sorted([f for f in os.listdir(self.clean_dir) if f.endswith(('png', 'jpg'))])

        # 确保 hazy 和 clean 文件夹中图片的数量一致
        assert len(self.hazy_images) == len(self.clean_images), "Hazy and clean images do not match in number."
      
    def __len__(self):
        # 返回图片数量
        return len(self.hazy_images)

    def __getitem__(self, index):
        # 根据索引获取对应的 hazy 和 clean 图像文件名
        hazy_image_name = self.hazy_images[index]
        clean_image_name = self.clean_images[index]

        # 读取模糊图像和清晰图像
        hazy_im = cv2.imread(os.path.join(self.hazy_dir, hazy_image_name))
        clean_im = cv2.imread(os.path.join(self.clean_dir, clean_image_name))
        
        # 检查图像是否正确读取
        if hazy_im is None:
            raise FileNotFoundError(f"Cannot load hazy image at {os.path.join(self.hazy_dir, hazy_image_name)}")
        if clean_im is None:
            raise FileNotFoundError(f"Cannot load clean image at {os.path.join(self.clean_dir, clean_image_name)}")

        # 转换颜色 BGR -> RGB
        hazy_im = hazy_im[:, :, ::-1]  # BGR to RGB
        clean_im = clean_im[:, :, ::-1]  # BGR to RGB
        
        # 统一调整图像大小
        target_size = (256, 256)  # 你可以根据需求设定目标大小
        hazy_im = cv2.resize(hazy_im, target_size)
        clean_im = cv2.resize(clean_im, target_size)

        # 归一化图像到 [0, 1] 范围
        hazy_im = np.float32(hazy_im) / 255.0
        clean_im = np.float32(clean_im) / 255.0

        # 创建样本字典
        sample = {'raw': hazy_im, 'reference': clean_im}    
        
        # 如果有定义 transform，应用到样本上
        if self.transform is not None:
            sample = self.transform(sample)

        return sample