"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt
import shutil
from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import os
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import PIL
import scipy.stats as st
import sys
import imageio
import skimage.io as io
import cv2

FLAGS = tf.app.flags.FLAGS

def transform(images):
  return np.array(images)/127.5 - 1.
def inverse_transform(images):
  return (images+1.)/2


def imread(path, is_grayscale=False):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  # import pdb 
  # pdb.set_trace()
  if is_grayscale:
    return io.imread(path, flatten=True).astype(float)
  else:
    return io.imread(path).astype(float)

    
def imsave(image, path):
  # import pdb 
  # pdb.set_trace()
  imsaved = (inverse_transform(image)).astype(float)
  return io.imsave(path, imsaved)

# def get_image(image_path,is_grayscale=False):
#   image = imread(image_path, is_grayscale)
#   # import pdb 
#   # pdb.set_trace()
#   #return transform(image)
#   return image/255


# 读取单张图片
def get_image_original(image_path,is_grayscale=False):
  image = io.imread(image_path, is_grayscale)
  image = image.astype(np.float32)
  if image.shape[-1] == 4:
    # 如果图像有四个通道，假设第四个通道为 alpha 通道，只保留前三个通道（RGB）
    image = image[:, :, :3]
  return image/255.0


# 获取白平衡结果
def get_images_wb(img):
    im_rgb = (img*255).astype(np.uint8)
    # if RGB
    R = np.sum(im_rgb[:, :, 0], axis=None)
    G = np.sum(im_rgb[:, :, 1], axis=None)
    B = np.sum(im_rgb[:, :, 2], axis=None)

    maxpix = max(R, G, B)
    ratio = np.array([maxpix / R, maxpix / G, maxpix / B])

    satLevel1 = 0.005 * ratio
    satLevel2 = 0.005 * ratio

    m, n, p = im_rgb.shape
    im_rgb_flat = np.zeros(shape=(p, m * n))
    for i in range(0, p):
        im_rgb_flat[i, :] = np.reshape(im_rgb[:, :, i], (1, m * n))

    wb = np.zeros(shape=im_rgb_flat.shape)
    for ch in range(p):
        q = [satLevel1[ch], 1 - satLevel2[ch]]
        tiles = np.quantile(im_rgb_flat[ch, :], q)
        temp = im_rgb_flat[ch, :]
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        wb[ch, :] = temp
        bottom = min(wb[ch, :])
        top = max(wb[ch, :])
        wb[ch, :] = (wb[ch, :] - bottom) * 255 / (top - bottom)

    outval = np.zeros(shape=im_rgb.shape)
    for i in range(p):
        outval[:, :, i] = np.reshape(wb[i, :], (m, n))
    
    outval = np.clip(outval/255.0, 0, 1)
    return outval


# 获取伽马校正结果
def get_images_gc(img, gamma=0.7):
  img_gc = np.power(img, gamma)
  img_gc = np.clip(img_gc, 0, 1)
  return img_gc


# 获取分块 (8,8) 直方图均衡化结果
def get_images_histeq(img):
  img = (img*255).astype(np.uint8)
  im_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
  clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
  el = clahe.apply(im_lab[:, :, 0])
  im_he = im_lab.copy()
  im_he[:, :, 0] = el
  img_histeq = cv2.cvtColor(im_he, cv2.COLOR_LAB2RGB)
  img_histeq = np.clip(img_histeq/255.0, 0, 1)
  return img_histeq


def prepare_data(sess, dataset, isRawImages=True):
  # import pdb 
  # pdb.set_trace()
  
  # 收集所有符合条件的文件（jpg、png、jpeg）
  data_dir = os.path.join(os.getcwd(), dataset)
  print("datadir_____:",data_dir)
  data = glob.glob(os.path.join(data_dir, "*.png"))
  data += glob.glob(os.path.join(data_dir, "*.jpg"))
  data += glob.glob(os.path.join(data_dir, "*.jpeg"))
  print("data:_____",len(data))
  # 退化图像需要计算 伽马校正 与 直方图均衡化 结果
  if isRawImages:

    # # 生成gc、wb、histeq
    # # 更新 白平衡、伽马校正 和 直方图均衡化 处理后的图像保存目录
    # wb_dir = os.path.join(os.getcwd(), f"{dataset}-wb")
    # if os.path.exists(wb_dir):
    #   shutil.rmtree(wb_dir)  # 删除目录及其所有内容
    # os.makedirs(wb_dir)  # 重新创建目录
    # gc_dir = os.path.join(os.getcwd(), f"{dataset}-gc")
    # if os.path.exists(gc_dir):
    #   shutil.rmtree(gc_dir)  # 删除目录及其所有内容
    # os.makedirs(gc_dir)  # 重新创建目录
    # histeq_dir = os.path.join(os.getcwd(), f"{dataset}-histeq")
    # if os.path.exists(histeq_dir):
    #   shutil.rmtree(histeq_dir)  # 删除目录及其所有内容
    # os.makedirs(histeq_dir)  # 重新创建目录
    # num = 0
    # # 对图像进行处理并保存
    # for file in data:
    #   num = num + 1
    #   print("开始处理第 ",num," 张图像的 WB & GC & Histeq 变换")
    #   # 读取图像
    #   img = get_image_original(file, is_grayscale=False)
    #   # 应用白平衡
    #   img_wb = get_images_wb(img)
    #   img_wb = np.array(img_wb)
    #   wb_filename = os.path.join(wb_dir, os.path.basename(file))
    #   plt.imsave(wb_filename, img_wb)
    #   # 应用伽马校正
    #   img_gc = get_images_gc(img, gamma=0.7)
    #   img_gc = np.array(img_gc)
    #   gc_filename = os.path.join(gc_dir, os.path.basename(file))
    #   plt.imsave(gc_filename, img_gc)
    #   # 应用直方图均衡化
    #   img_histeq = get_images_histeq(img)
    #   img_histeq = np.array(img_histeq)
    #   histeq_filename = os.path.join(histeq_dir, os.path.basename(file))
    #   plt.imsave(histeq_filename, img_histeq)
    # # 生成gc、wb、histeq
    
    # 白平衡 收集所有符合条件的文件（jpg、png、jpeg）
    data_wb_dir = os.path.join(os.getcwd(), f"{dataset}-wb")
    print("data_wb_dir________:",data_wb_dir)
    wb_data = glob.glob(os.path.join(data_wb_dir, "*.png"))
    wb_data += glob.glob(os.path.join(data_wb_dir, "*.jpg"))
    wb_data += glob.glob(os.path.join(data_wb_dir, "*.jpeg"))
    print("wb_data:_____",len(wb_data))
    # 伽马校正 收集所有符合条件的文件（jpg、png、jpeg）
    data_gc_dir = os.path.join(os.getcwd(), f"{dataset}-gc")
    gc_data = glob.glob(os.path.join(data_gc_dir, "*.png"))
    gc_data += glob.glob(os.path.join(data_gc_dir, "*.jpg"))
    gc_data += glob.glob(os.path.join(data_gc_dir, "*.jpeg"))
    print("gc_data:_____",len(gc_data))
    # 直方图均衡化 收集所有符合条件的文件（jpg、png、jpeg）
    data_histeq_dir = os.path.join(os.getcwd(), f"{dataset}-histeq")
    histeq_data = glob.glob(os.path.join(data_histeq_dir, "*.png"))
    histeq_data += glob.glob(os.path.join(data_histeq_dir, "*.jpg"))
    histeq_data += glob.glob(os.path.join(data_histeq_dir, "*.jpeg"))
    print("histeq_data:_____",len(histeq_data))

    # 使用完整路径进行排序（考虑路径和扩展名）
    sorted_data = sorted(data, key=lambda x: os.path.abspath(x))
    sorted_wb_data = sorted(wb_data, key=lambda x: os.path.abspath(x))
    sorted_gc_data = sorted(gc_data, key=lambda x: os.path.abspath(x))
    sorted_histeq_data = sorted(histeq_data, key=lambda x: os.path.abspath(x))  
    
    return sorted_data, sorted_wb_data, sorted_gc_data, sorted_histeq_data

  else: # 是参考数据
    # 使用完整路径进行排序（考虑路径和扩展名）
    sorted_data = sorted(data, key=lambda x: os.path.abspath(x))
    
    return sorted_data
  
  
# 读取单张图片，裁剪大小，转换类型
def get_image_raw(image_path, is_grayscale=False):
  
  image = io.imread(image_path, is_grayscale)
  # 检查图像通道数
  if image.shape[-1] == 4:
    # 如果图像有四个通道，假设第四个通道为 alpha 通道，只保留前三个通道（RGB）
    image = image[:, :, :3]
    
  image = np.asarray(np.float32(image)/255)
  k = 256
  # 生成 1 个介于 0（包含）到 330/490（不包含）之间的随机整数
  x_offset = np.random.randint(low=0, high=image.shape[0] - k + 1, size=1)[0]
  y_offset = np.random.randint(low=0, high=image.shape[1] - k + 1, size=1)[0]
  # print(image_path, "------ image.shape[0]:",image.shape[0]," image.shape[1]:",image.shape[1], "x_offset:", x_offset, " y_offset:", y_offset)
          
  # 检查图像尺寸是否足够大以支持裁剪操作
  if image.shape[0] >= k + x_offset and image.shape[1] >= k + y_offset:
    # 对数据随机裁剪出 高128*宽128 的区域
    cropped_batch = image[x_offset:x_offset+k, y_offset:y_offset+k, :]
    # print("raw:",image.shape[0],image.shape[1],image_path,x_offset,y_offset)
    return cropped_batch,x_offset,y_offset
      
  # 不够大
  else:
    # 处理图像尺寸过小的情况
    if image.shape[0]-k < 0:
      print(image.shape)
      x_offset = 0
    if image.shape[1]-k < 0:
      print(image.shape)
      y_offset = 0
    
    # 对数据随机裁剪出 高128*宽128 的区域
    cropped_batch = image[x_offset:x_offset+k, y_offset:y_offset+k, :]
    print("Raw Too small!!!")
    return cropped_batch,x_offset,y_offset
    # return np.array(cropped_batch / 255.0).astype(np.float32)
  

# 读取单张图片的参考图像/伽马校正/直方图均衡化，裁剪大小，转换类型
def get_image_reference_wb_gc_histeq(image_path, x_offset=0, y_offset=0, is_grayscale=False):
  image = io.imread(image_path, is_grayscale)
  # 检查图像通道数
  if image.shape[-1] == 4:
    # 如果图像有四个通道，假设第四个通道为 alpha 通道，只保留前三个通道（RGB）
    image = image[:, :, :3]
  image = np.asarray(np.float32(image)/255)
  # print(image_path, "------ refimage.shape[0]:",image.shape[0]," refimage.shape[1]:",image.shape[1], "x_offset:", x_offset, " y_offset:", y_offset)

  k = 256
  # 检查图像尺寸是否足够大以支持裁剪操作
  if image.shape[0] >= k + x_offset and image.shape[1] >= k + y_offset:
  # 对数据随机裁剪出 高128*宽128 的区域
    cropped_batch = image[x_offset:x_offset+k, y_offset:y_offset+k, :]
    # print(image_path,image.shape[0]," ",image.shape[1],"!!!!!",k,k + x_offset,k + y_offset)
    return cropped_batch
  else:
    print(image_path,image.shape[0]," ",image.shape[1],"Reference Too small",k,k + x_offset,k + y_offset)
    

def patch_merging_with_concat(x):
    # 分别提取 2x2 patch 中的四个位置的像素块
    x1 = x[:, 0::2, 0::2, :]  # 提取 (0, 0) 位置
    x2 = x[:, 0::2, 1::2, :]  # 提取 (0, 1) 位置
    x3 = x[:, 1::2, 0::2, :]  # 提取 (1, 0) 位置
    x4 = x[:, 1::2, 1::2, :]  # 提取 (1, 1) 位置
    
    # 将四个提取的张量在通道维度上拼接
    x_merged = tf.concat([x1, x2, x3, x4], axis=-1)
    
    return x_merged



def get_lable(image_path,is_grayscale=False):
  image = io.imread(image_path, is_grayscale)
  return image/255.
def imsave_lable(image, path):
  return io.imsave(path, image*255)

def loss_gradient_difference(true, generated):
   true_x_shifted_right = true[:,1:,:,:]
   true_x_shifted_left = true[:,:-1,:,:]
   true_x_gradient = tf.abs(true_x_shifted_right - true_x_shifted_left)

   generated_x_shifted_right = generated[:,1:,:,:]
   generated_x_shifted_left = generated[:,:-1,:,:]
   generated_x_gradient = tf.abs(generated_x_shifted_right - generated_x_shifted_left)

   loss_x_gradient = tf.reduce_mean(tf.square(true_x_gradient - generated_x_gradient))

   true_y_shifted_right = true[:,:,1:,:]
   true_y_shifted_left = true[:,:,:-1,:]
   true_y_gradient = tf.abs(true_y_shifted_right - true_y_shifted_left)

   generated_y_shifted_right = generated[:,:,1:,:]
   generated_y_shifted_left = generated[:,:,:-1,:]
   generated_y_gradient = tf.abs(generated_y_shifted_right - generated_y_shifted_left)
    
   loss_y_gradient = tf.reduce_mean(tf.square(true_y_gradient - generated_y_gradient))

   loss = loss_x_gradient + loss_y_gradient
   return loss

def gredient(x):
  # _,h,w,_=x.shape
  # g_x = np.zeros(x.shape)
  # g_y = np.zeros(x.shape)
  g_x = x[:,0:-1,:,:]-x[:,1:,:,:]
  # g_x[:,-1,:,:] = x[:,-1,:,:]
  g_y = x[:,:,0:-1,:]-x[:,:,1:,:]
  # g_y[:,:,-1,:] = x[:,:,-1,:]
  g = tf.reduce_mean(tf.abs(g_x))+tf.reduce_mean(tf.abs(g_y))
  return g


def random_crop_and_flip_1(batch_data, x_offset, y_offset,w_size,h_size):
    
    cropped_batch = np.zeros(len(batch_data) * w_size * h_size * 1).reshape(
                 len(batch_data), w_size, h_size, 1)
    
    for i in range(len(batch_data)):
        #x_offset = np.random.randint(low=0, height=2*padding_size, size=1)[0]
        #x_offset = np.random.randint(low=0, height=2 * padding_size, size=1)[0]
        #import pdb  
        #pdb.set_trace()
        cropped_batch[i, :,:,:] = batch_data[i, x_offset:x_offset+w_size,y_offset:y_offset+h_size,:]
        #cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
        
    return cropped_batch

# 对输入的批量数据（bstch_size个）裁剪出指定大小的区域，并返回裁剪后的结果。
def tensor_random_crop_and_flip_3(batch_data, batch_size,x_offset, y_offset, h_size,w_size):
    # import pdb  
    # pdb.set_trace()  
    # cropped_batch = tf.zeros([1, 64, 64, 3])  
    # cropped_batch = tf.zeros([batch_size, w_size, h_size, 3])
    
    # for i in range(batch_size):
        #x_offset = np.random.randint(low=0, height=2*padding_size, size=1)[0]
        #x_offset = np.random.randint(low=0, height=2 * padding_size, size=1)[0]
        #import pdb  
        #pdb.set_trace()cropped_batch[i, :,:,:] = 
        
        #cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
        
    return batch_data[:, x_offset:x_offset+h_size,y_offset:y_offset+w_size,:] # 批量大小，高度，宽度，通道数（RGB为3）

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter


def MaxMinNormalization(x):
  x = (x - x.min()) / (x.max() - x.min())
  return x