from utils import ( 
  imsave,
  prepare_data
)
# wait for add to evaluate by metrics
# from ssim import *

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf1
import tensorflow.compat.v1 as tf
import scipy.io as scio
from ops import *
from ssim import *
from PIL import Image 
# local library
import rgb_lab_formulation as Conv_img
class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=256,
               image_width=256,
               label_height=256, 
               label_width=256,
               batch_size=1,
               counter=1,
               c_dim=3, 
               checkpoint_dir=None, 
               test_image_name = None,
               id = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.9
    self.test_image_name = test_image_name
    self.id = id
    self.counter = counter
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.new_height=0
    self.new_width=0
    self.new_height_half=0 
    self.new_width_half=0
    self.new_height_half_half=0
    self.new_width_half_half=0  
    image_test =  get_image_original(self.test_image_name,is_grayscale=False)
    shape = image_test.shape
    RGB=Image.fromarray(np.uint8(image_test*255))
    RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB1)/255)
    shape = image_test.shape
    self.new_height=shape[0]
    self.new_width=shape[1]

    self.build_model()
    # self.print_model_params()  # 添加这行以打印参数量
    

  def build_model(self):
    self.images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.images_wb = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_gc = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.images_histeq = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_histeq')
    self.pred_h = self.model()
    self.saver = tf.compat.v1.train.Saver()
  
  
  # # 获取参数量
  # def print_model_params(self):
  #       # 获取所有可训练的变量
  #       trainable_params = tf.compat.v1.trainable_variables()
  #       total_params = 0
  #       for var in trainable_params:
  #           shape = var.get_shape()
  #           num_params = np.prod(shape)  # 计算每个变量的参数数量
  #           total_params += num_params
  #           # print(f'{var.name}: {shape} -> {num_params} parameters')
  #       print(f'Total parameters: {total_params}')


  def train(self, config):

    # 获取 测试图像
    image_test0 =  get_image_original(self.test_image_name,is_grayscale=False)
    # image_test0 = cv2.resize(image_test0, (256,256))
    original_shape = image_test0.shape
    original_size = (original_shape[1], original_shape[0])
    # 使图像宽高能被8整除
    RGB=Image.fromarray(np.uint8(image_test0*255))
    RGB1=RGB.resize(((original_shape[1]//8-0)*8,(original_shape[0]//8-0)*8))
    image_test = np.asarray(np.float32(RGB1)/255)
    shape = image_test.shape
    # 为图像增加一个新的维度，使其能够与批处理张量兼容
    expand_test = image_test[np.newaxis,:,:,:]
    # 需要增加的批次量，并加上
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)
    
    # 获取测试图像的 白平衡图像
    image_test_wb = get_images_wb(image_test)
    # image_test_wb = cv2.resize(image_test_wb, (256,256))
    shape_wb = image_test_wb.shape
    # 使图像宽高能被8整除
    RGB_wb=Image.fromarray(np.uint8(image_test_wb*255))
    RGB1_wb=RGB_wb.resize(((shape_wb[1]//8-0)*8,(shape_wb[0]//8-0)*8))
    image_test_wb = np.asarray(np.float32(RGB1_wb)/255)
    shape_wb = image_test_wb.shape
    # 为图像增加一个新的维度，使其能够与批处理张量兼容
    expand_test_wb = image_test_wb[np.newaxis,:,:,:]
    # 需要增加的批次量，并加上
    expand_zero_wb = np.zeros([self.batch_size-1,shape_wb[0],shape_wb[1],shape_wb[2]])
    batch_test_image_wb = np.append(expand_test_wb,expand_zero_wb,axis = 0)
    
    # 获取测试图像的 伽马校正图像
    image_test_gc = get_images_gc(image_test, gamma=0.7)
    # image_test_gc = cv2.resize(image_test_gc, (256,256))
    shape_gc = image_test_gc.shape
    # 使图像宽高能被8整除
    RGB_gc=Image.fromarray(np.uint8(image_test_gc*255))
    RGB1_gc=RGB_gc.resize(((shape_gc[1]//8-0)*8,(shape_gc[0]//8-0)*8))
    image_test_gc = np.asarray(np.float32(RGB1_gc)/255)
    shape_gc = image_test_gc.shape
    # 为图像增加一个新的维度，使其能够与批处理张量兼容
    expand_test_gc = image_test_gc[np.newaxis,:,:,:]
    # 需要增加的批次量，并加上
    expand_zero_gc = np.zeros([self.batch_size-1,shape_gc[0],shape_gc[1],shape_gc[2]])
    batch_test_image_gc = np.append(expand_test_gc,expand_zero_gc,axis = 0)
    
    # 获取测试图像的 直方图均衡化图像
    image_test_histeq = get_images_histeq(image_test)
    # image_test_histeq = cv2.resize(image_test_histeq, (256,256))
    shape_histeq = image_test_histeq.shape
    # 使图像宽高能被8整除
    RGB_histeq=Image.fromarray(np.uint8(image_test_histeq*255))
    RGB1_histeq=RGB_histeq.resize(((shape_histeq[1]//8-0)*8,(shape_histeq[0]//8-0)*8))
    image_test_histeq = np.asarray(np.float32(RGB1_histeq)/255)
    shape_histeq = image_test_histeq.shape
    # 为图像增加一个新的维度，使其能够与批处理张量兼容
    expand_test_histeq = image_test_histeq[np.newaxis,:,:,:]
    # 需要增加的批次量，并加上
    expand_zero_histeq = np.zeros([self.batch_size-1,shape_histeq[0],shape_histeq[1],shape_histeq[2]])
    batch_test_image_histeq = np.append(expand_test_histeq,expand_zero_histeq,axis = 0)    
    
    tf.compat.v1.global_variables_initializer().run()
    
    self.counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    start_time = time.time()
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image, self.images_wb:batch_test_image_wb, self.images_gc: batch_test_image_gc, self.images_histeq: batch_test_image_histeq})
    all_time = time.time()
    final_time=all_time - start_time
    print(final_time)    
    
    _,h ,w , c = result_h.shape
    # print(result_h.shape)
    for id in range(0,1):
      final_enhanced_img = np.clip(result_h[id], 0, 1).reshape(h , w , 3)
      # 保存路径
      imgname_cor = self.test_image_name.split('.')[0] # 图片名称，提取前缀
      image_path1 = os.path.join(os.getcwd(), imgname_cor+'_out.'+self.test_image_name.split('.')[1])
      # print("imgname_cor: ", imgname_cor)
      # print("image_path1: ", image_path1)

      # # 若直接保存
      # plt.imsave(image_path1, final_enhanced_img)
      
      # resize回 原大小 保存
      result_h0 = (final_enhanced_img * 255).astype(np.uint8)
      a=Image.fromarray(result_h0)
      b=a.resize(original_size)
      original_size_enhanced = np.array(b)
      io.imsave(image_path1, original_size_enhanced) 
      print(str(self.id)+": "+image_path1)


  # 网络结构
  def model(self):
    with tf.compat.v1.variable_scope("fusion_branch") as scope1: 
    # 创建了一个 TensorFlow 变量作用域（Variable Scope）
    # 这个变量作用域内创建的 所有变量 都会被自动命名为 "fusion_branch/变量名"，以便与其他变量作用域中的同名变量区分开来

# first RGB encoder：输出 conv1_end1_RGB 3-64-128
      # # 边缘检测模块
      # conv1_RGB_0_Refine1 = tf.nn.relu(conv2d(self.images, 3, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGB_0_Refine1")) 
      # conv1_RGB_0_Refine2 = tf.nn.relu(conv2d(conv1_RGB_0_Refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGB_0_Refine2")) 
      # conv1_RGB_res0 = conv2d(conv1_RGB_0_Refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGB_res0") 
      # conv1_RGB_edge = tf.add(conv1_RGB_res0,conv1_RGB_0_Refine1)
      # conv1_RGB_edge0 = tf.add(conv1_RGB_edge,conv1_RGB_0_Refine2)# 与解码器残差连接
      
      CM_cat_RGB = tf.concat(axis = 3, values = [self.images, self.images_wb, self.images_gc, self.images_histeq]) # axis=3表示第四个维度（通道）进行拼接
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅
      conv1_RGB_0 = tf.nn.relu(conv2d(CM_cat_RGB, 12, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGB_0"))
      conv1_RGB_0_Refine1 = tf.nn.relu(conv2d(conv1_RGB_0, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGB_0_Refine1")) 
      conv1_RGB_0_Refine2 = tf.nn.relu(conv2d(conv1_RGB_0_Refine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGB_0_Refine2")) 
      conv1_RGB_res0 = conv2d(conv1_RGB_0_Refine2, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGB_res0") 
      conv1_RGB = tf.add(conv1_RGB_res0,conv1_RGB_0)# 与解码器残差连接
            
      # 多尺度卷积：64*H*W 变为 128*H/2*W/2，拼接后变为 RGB_mt1 384*H/2*W/2
      # 1*1
      ### conv1_mt1_1_RGB = tf.nn.relu(conv2d(conv1_RGB, 128,256,k_h=1, k_w=1, d_h=2, d_w=2,name="conv1_mt1_1_RGB"))
      conv1_mt1_1_RGB = tf.nn.relu(conv2d(conv1_RGB, 16, 32, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_RGB"))
      # conv1_mt1_1_1_RGB = tf.nn.relu(conv2d(conv1_mt1_1_RGB, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_1_RGB"))
      # conv1_mt1_1_2_RGB = tf.nn.relu(conv2d(conv1_mt1_1_1_RGB, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_2_RGB"))
      # conv1_mt1_1_3_RGB = conv2d(conv1_mt1_1_2_RGB, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_3_RGB")
      # conv1_mt1_1_4_RGB = tf.add(conv1_mt1_1_RGB, conv1_mt1_1_3_RGB)
      # 3*3
      conv1_mt1_2_RGB = tf.nn.relu(conv2d(conv1_RGB, 16, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_RGB"))
      # conv1_mt1_2_1_RGB = tf.nn.relu(conv2d(conv1_mt1_2_RGB, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_1_RGB"))
      # conv1_mt1_2_2_RGB = tf.nn.relu(conv2d(conv1_mt1_2_1_RGB, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_2_RGB"))
      # conv1_mt1_2_3_RGB = conv2d(conv1_mt1_2_2_RGB, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_3_RGB")
      # conv1_mt1_2_4_RGB = tf.add(conv1_mt1_2_RGB, conv1_mt1_2_3_RGB)
      # 5*5
      conv1_mt1_3_RGB = tf.nn.relu(conv2d(conv1_RGB, 16, 32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_RGB"))
      # conv1_mt1_3_1_RGB = tf.nn.relu(conv2d(conv1_mt1_3_RGB, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_1_RGB"))
      # conv1_mt1_3_2_RGB = tf.nn.relu(conv2d(conv1_mt1_3_1_RGB, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_2_RGB"))
      # conv1_mt1_3_3_RGB = conv2d(conv1_mt1_3_2_RGB, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_3_RGB")
      # conv1_mt1_3_4_RGB = tf.add(conv1_mt1_3_RGB, conv1_mt1_3_3_RGB)
      
      RGB_mt1= tf.concat(axis = 3, values = [conv1_mt1_1_RGB,conv1_mt1_2_RGB,conv1_mt1_3_RGB]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=128*3=384)
      RGB_mt1_SCA = self.Channel_Space_Attention_layer(RGB_mt1, out_dim=96, ratio=1, layer_name="RGB_mt1_SCA")

      # 多尺度卷积
      
      # SCA   缩放比例 ratio
      # RGB_mt1_SCA = self.Channel_Space_Attention_layer(RGB_mt1, out_dim=384, ratio=8, layer_name="RGB_mt1_SCA")
      # # RGB_mt1_SCA_toAdd = tf.nn.relu(conv2d(RGB_mt1_SCA,768,768,k_h=3, k_w=3, d_h=1, d_w=1,name="RGB_mt1_SCA_toAdd")) 
      # # SCA结果与SCA输入残差连接 得到 RGB_SCA_1：(batch_size, height=H/2, width=W/2, channels=128*3=384)
      # RGB_SCA_1 = tf.add(RGB_mt1, RGB_mt1_SCA)
      
      # 结束conv 得到 conv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=128)
      conv1_end1_RGB_toRefine = tf.nn.relu(conv2d(RGB_mt1_SCA, 96, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_RGB_toRefine"))
      conv1_end1_RGB_Refine1 = tf.nn.relu(conv2d(conv1_end1_RGB_toRefine, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_RGB_Refine1"))
      conv1_end1_RGB_Refine2 = tf.nn.relu(conv2d(conv1_end1_RGB_Refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_RGB_Refine2"))
      conv1_end1_RGB_res0 = conv2d(conv1_end1_RGB_Refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_RGB_res0") 
      conv1_end1_RGB_res1 = tf.add(conv1_end1_RGB_toRefine, conv1_end1_RGB_res0) # 与解码器特征融合
      conv1_end1_RGB_res = tf.nn.max_pool(conv1_end1_RGB_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Second RGB encoder：输出 conv2_end2_RGB 128-256
      RGB_input_scale2 =  tf.nn.max_pool(CM_cat_RGB, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      conv1_RGBscale2_0 = tf.nn.relu(conv2d(RGB_input_scale2, 12, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGBscale2_0"))
      conv1_RGBscale2_0_Refine1 = tf.nn.relu(conv2d(conv1_RGBscale2_0, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGBscale2_0_Refine1")) 
      conv1_RGBscale2_0_Refine2 = tf.nn.relu(conv2d(conv1_RGBscale2_0_Refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGBscale2_0_Refine2")) 
      conv1_RGBscale2_res0 = conv2d(conv1_RGBscale2_0_Refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGBscale2_res0") 
      conv1_RGBscale2 = tf.add(conv1_RGBscale2_res0,conv1_RGBscale2_0)# 与解码器残差连接
      RGB_input2 = tf.concat(axis = 3, values = [conv1_end1_RGB_res, conv1_RGBscale2])
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅      
      # 多尺度卷积：128*H/2*W/2 变为 256*H/2*W/2，拼接后变为 RGB_mt2 768*H/2*W/2
      # 1*1
      conv2_mt2_1_RGB = tf.nn.relu(conv2d(RGB_input2, 64, 64, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_RGB"))
      # conv2_mt2_1_1_RGB = tf.nn.relu(conv2d(conv2_mt2_1_RGB, 256, 256, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_1_RGB"))
      # conv2_mt2_1_2_RGB = tf.nn.relu(conv2d(conv2_mt2_1_1_RGB, 256, 256, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_2_RGB"))
      # conv2_mt2_1_3_RGB = tf.nn.relu(conv2d(conv2_mt2_1_2_RGB, 256, 256, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_3_RGB"))
      # conv2_mt2_1_4_RGB = tf.add(conv2_mt2_1_RGB, conv2_mt2_1_3_RGB)
      # 3*3
      conv2_mt2_2_RGB = tf.nn.relu(conv2d(RGB_input2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_RGB"))
      # conv2_mt2_2_1_RGB = tf.nn.relu(conv2d(conv2_mt2_2_RGB, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_1_RGB"))
      # conv2_mt2_2_2_RGB = tf.nn.relu(conv2d(conv2_mt2_2_1_RGB, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_2_RGB"))
      # conv2_mt2_2_3_RGB = conv2d(conv2_mt2_2_2_RGB, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_3_RGB")
      # conv2_mt2_2_4_RGB = tf.add(conv2_mt2_2_RGB, conv2_mt2_2_3_RGB)
      # 5*5
      conv2_mt2_3_RGB = tf.nn.relu(conv2d(RGB_input2, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_RGB"))
      # conv2_mt2_3_1_RGB = tf.nn.relu(conv2d(conv2_mt2_3_RGB, 256, 256, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_1_RGB"))
      # conv2_mt2_3_2_RGB = tf.nn.relu(conv2d(conv2_mt2_3_1_RGB, 256, 256, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_2_RGB"))
      # conv2_mt2_3_3_RGB = conv2d(conv2_mt2_3_2_RGB, 256, 256, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_3_RGB")
      # conv2_mt2_3_4_RGB = tf.add(conv2_mt2_3_RGB, conv2_mt2_3_3_RGB)
      
      RGB_mt2= tf.concat(axis = 3, values = [conv2_mt2_1_RGB, conv2_mt2_2_RGB, conv2_mt2_3_RGB]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=256*3=768)
      RGB_mt2_SCA = self.Channel_Space_Attention_layer(RGB_mt2, out_dim=192, ratio=1, layer_name="RGB_mt2_SCA")

      # 多尺度卷积
      
      # SCA    缩放比例 ratio
      # RGB_mt2_SCA = self.Channel_Space_Attention_layer(RGB_mt2, out_dim=768, ratio=8, layer_name="RGB_mt2_SCA")     
      # # RGB_mt2_SCA_toAdd = tf.nn.relu(RGB_mt2_SCA)
      # # RGB_mt2_SCA_toAdd = tf.nn.relu(conv2d(RGB_mt2_SCA,1536,1536,k_h=3, k_w=3, d_h=1, d_w=1,name="RGB_mt2_SCA_toAdd")) 
      # # SCA结果与SCA输入残差连接 得到 RGB_SCA_2：(batch_size, height=H/2, width=W/2, channels=256*3=768)
      # RGB_SCA_2 = tf.add(RGB_mt2, RGB_mt2_SCA)
      
      # 结束conv 得到 conv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=256)
      conv2_end2_RGB_toRefine = tf.nn.relu(conv2d(RGB_mt2_SCA, 192, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_RGB_toRefine"))
      conv2_end2_RGB_refine1 = tf.nn.relu(conv2d(conv2_end2_RGB_toRefine, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_RGB_refine1"))
      conv2_end2_RGB_refine2 = tf.nn.relu(conv2d(conv2_end2_RGB_refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_RGB_refine2"))
      conv2_end2_RGB_res0 = conv2d(conv2_end2_RGB_refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_RGB_res0")
      conv2_end2_RGB_res1 = tf.add(conv2_end2_RGB_toRefine, conv2_end2_RGB_res0)
      conv2_end2_RGB_res = tf.nn.max_pool(conv2_end2_RGB_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

         
# Third RGB encoder：
      RGB_input_scale3 =  tf.nn.max_pool(CM_cat_RGB, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
      conv1_RGBscale3_0 = tf.nn.relu(conv2d(RGB_input_scale3, 12, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGBscale3_0"))
      conv1_RGBscale3_0_Refine1 = tf.nn.relu(conv2d(conv1_RGBscale3_0, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGBscale3_0_Refine1")) 
      conv1_RGBscale3_0_Refine2 = tf.nn.relu(conv2d(conv1_RGBscale3_0_Refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGBscale3_0_Refine2")) 
      conv1_RGBscale3_res0 = conv2d(conv1_RGBscale3_0_Refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_RGBscale3_res0") 
      conv1_RGBscale3 = tf.add(conv1_RGBscale3_res0,conv1_RGBscale3_0)# 与解码器残差连接
      RGB_input3 = tf.concat(axis = 3, values = [conv2_end2_RGB_res, conv1_RGBscale3])
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅      
      # 多尺度卷积：
      # 1*1
      conv3_mt3_1_RGB = tf.nn.relu(conv2d(RGB_input3, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_RGB"))
      # conv3_mt3_1_1_RGB = tf.nn.relu(conv2d(conv3_mt3_1_RGB, 512, 512, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_1_RGB"))
      # conv3_mt3_1_2_RGB = tf.nn.relu(conv2d(conv3_mt3_1_1_RGB, 512, 512, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_2_RGB"))
      # conv3_mt3_1_3_RGB = conv2d(conv3_mt3_1_2_RGB, 512, 512, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_3_RGB")
      # conv3_mt3_1_4_RGB = tf.add(conv3_mt3_1_RGB, conv3_mt3_1_3_RGB)
      # 3*3
      conv3_mt3_2_RGB = tf.nn.relu(conv2d(RGB_input3, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_RGB"))
      # conv3_mt3_2_1_RGB = tf.nn.relu(conv2d(conv3_mt3_2_RGB, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_1_RGB"))
      # conv3_mt3_2_2_RGB = tf.nn.relu(conv2d(conv3_mt3_2_1_RGB, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_2_RGB"))
      # conv3_mt3_2_3_RGB = conv2d(conv3_mt3_2_2_RGB, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_3_RGB")
      # conv3_mt3_2_4_RGB = tf.add(conv3_mt3_2_RGB, conv3_mt3_2_3_RGB)
      # 5*5
      conv3_mt3_3_RGB = tf.nn.relu(conv2d(RGB_input3, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_RGB"))
      # conv3_mt3_3_1_RGB = tf.nn.relu(conv2d(conv3_mt3_3_RGB, 512, 512, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_1_RGB"))
      # conv3_mt3_3_2_RGB = tf.nn.relu(conv2d(conv3_mt3_3_1_RGB, 512, 512, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_2_RGB"))
      # conv3_mt3_3_3_RGB = conv2d(conv3_mt3_3_2_RGB, 512, 512, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_3_RGB")
      # conv3_mt3_3_4_RGB = tf.add(conv3_mt3_3_RGB, conv3_mt3_3_3_RGB)
      
      RGB_mt3 = tf.concat(axis = 3, values = [conv3_mt3_1_RGB, conv3_mt3_2_RGB, conv3_mt3_3_RGB]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=256*3=768)
      RGB_mt3_SCA = self.Channel_Space_Attention_layer(RGB_mt3, out_dim=384, ratio=1, layer_name="RGB_mt3_SCA")
      # 多尺度卷积
      
      # SCA    缩放比例 ratio
      # RGB_mt3_SCA = self.Channel_Space_Attention_layer(RGB_mt3, out_dim=1536, ratio=8, layer_name="RGB_mt3_SCA")     
      # # RGB_mt3_SCA_toAdd = tf.nn.relu(RGB_mt2_SCA)
      # # RGB_mt2_SCA_toAdd = tf.nn.relu(conv2d(RGB_mt2_SCA,1536,1536,k_h=3, k_w=3, d_h=1, d_w=1,name="RGB_mt2_SCA_toAdd")) 
      # # SCA结果与SCA输入残差连接 得到 RGB_SCA_2：(batch_size, height=H/2, width=W/2, channels=256*3=768)
      # RGB_SCA_3 = tf.add(RGB_mt3, RGB_mt3_SCA)
      
      # 结束conv 得到 conv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=256)
      conv3_end3_RGB_toRefine = tf.nn.relu(conv2d(RGB_mt3_SCA, 384, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_RGB_toRefine"))
      conv3_end3_RGB_refine1 = tf.nn.relu(conv2d(conv3_end3_RGB_toRefine, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_RGB_refine1"))
      conv3_end3_RGB_refine2 = tf.nn.relu(conv2d(conv3_end3_RGB_refine1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_RGB_refine2"))
      conv3_end3_RGB_res0 = conv2d(conv3_end3_RGB_refine2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_RGB_res0")
      conv3_end3_RGB_res1 = tf.add(conv3_end3_RGB_toRefine, conv3_end3_RGB_res0)
      conv3_end3_RGB_res = tf.nn.max_pool(conv3_end3_RGB_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
         
      fusion_res_SCA_RGB = self.Channel_Space_Attention_layer(conv3_end3_RGB_res, out_dim=128, ratio=1, layer_name="fusion_res_SCA_RGB")
 
  # Zero decoder
    # pre 精细化 512
      de_conv_0_re0 = tf.nn.relu(conv2d(fusion_res_SCA_RGB, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_0_re0"))
      de_conv_0_re1 = tf.nn.relu(conv2d(de_conv_0_re0, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_0_re1"))
      de_conv_0_re2 = tf.nn.relu(conv2d(de_conv_0_re1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_0_re2"))
      de_conv_0_re3 = conv2d(de_conv_0_re2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_0_re3")
      de_conv_0_re1_add = tf.add(de_conv_0_re3, de_conv_0_re0)
      
      # 获取置信度图 Confidence Map
      CM_cat = tf.concat(axis = 3, values = [self.images, self.images_wb, self.images_gc, self.images_histeq]) # axis=3表示第四个维度（通道）进行拼接
      CM1_pool = tf.nn.max_pool(CM_cat, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
      CM1_conv1 = tf.nn.tanh(conv2d(CM1_pool, 12, 128, k_h=7, k_w=7, d_h=1, d_w=1,name="CM1_conv1"))
      CM1_conv2 = tf.nn.tanh(conv2d(CM1_conv1, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="CM1_conv2"))
      CM1_conv3 = tf.nn.tanh(conv2d(CM1_conv2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="CM1_conv3"))
      CM1_conv4 = tf.nn.tanh(conv2d(CM1_conv3, 128, 64, k_h=1, k_w=1, d_h=1, d_w=1,name="CM1_conv4"))
      CM2_conv1 = tf.nn.tanh(conv2d(CM1_conv4, 64, 64, k_h=7, k_w=7, d_h=1, d_w=1,name="CM2_conv1"))
      CM2_conv2 = tf.nn.tanh(conv2d(CM2_conv1, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1,name="CM2_conv2"))
      CM2_conv3 = tf.nn.tanh(conv2d(CM2_conv2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="CM2_conv3"))
      CM2_conv4 = tf.nn.sigmoid(conv2d(CM2_conv3, 64, 3, k_h=3, k_w=3, d_h=1, d_w=1,name="CM2_conv4"))
      out_wb, out_gc, out_histeq = tf.split(CM2_conv4, num_or_size_splits=[1, 1, 1], axis=-1)
      
      # 结果乘以置信度图获得最终结果
      de_conv_0_re1_add1 = tf.multiply(de_conv_0_re1_add, out_wb) + tf.multiply(de_conv_0_re1_add, out_gc) + tf.multiply(de_conv_0_re1_add, out_histeq)
      

# first decoder
      # 亚像素卷积上采样
      de_conv_1_up = tf.nn.relu(conv2d(de_conv_0_re1_add1, 128, 512, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_1_up"))
      de_conv_1_up2 = conv2d(de_conv_1_up, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_1_up2")
      de_conv_1_up3 = tf.nn.depth_to_space(de_conv_1_up2, block_size=2) # 通道数512/4=128
      # 残差连接第二层，融合特征 channel 128 concat 64  -> 192 -> 64
      # de_conv_1_tocat = tf.concat(axis = 3, values = [conv2_end2_RGB_res, conv2_end2_HSV_res, conv2_end2_LAB_res]) # axis=3表示第四个维度进行拼接
      de_conv_1_cat = tf.concat(axis = 3, values = [de_conv_1_up3, conv2_end2_RGB_res]) # axis=3表示第四个维度进行拼接
      toconcat_end_SCA_1 = self.Channel_Space_Attention_layer(de_conv_1_cat, out_dim=192, ratio=1, layer_name="toconcat_end_SCA_1")

      
      # 结束conv 得到 deconv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=128)
      deconv1_end1_RGB_toRefine = tf.nn.relu(conv2d(toconcat_end_SCA_1, 192, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_RGB_toRefine"))
      deconv1_end1_RGB_Refine1 = tf.nn.relu(conv2d(deconv1_end1_RGB_toRefine, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_RGB_Refine1"))
      deconv1_end1_RGB_Refine2 = tf.nn.relu(conv2d(deconv1_end1_RGB_Refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_RGB_Refine2"))
      deconv1_end1_RGB_res0 = conv2d(deconv1_end1_RGB_Refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_RGB_res0") 
      deconv1_end1_RGB_res1 = tf.add(deconv1_end1_RGB_res0,deconv1_end1_RGB_toRefine) 
      deconv1_end1_RGB_toRefine1 = tf.nn.relu(conv2d(deconv1_end1_RGB_res1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_RGB_toRefine1"))
      deconv1_end1_RGB_Refine3 = tf.nn.relu(conv2d(deconv1_end1_RGB_toRefine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_RGB_Refine3"))
      deconv1_end1_RGB_Refine4 = tf.nn.relu(conv2d(deconv1_end1_RGB_Refine3, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_RGB_Refine4"))
      deconv1_end1_RGB_res2 = conv2d(deconv1_end1_RGB_Refine4, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_RGB_res2") 
      deconv1_end1_RGB_res = tf.add(deconv1_end1_RGB_res2,deconv1_end1_RGB_toRefine1) 
      


# Second decoder
      # 亚像素卷积上采样
      de_conv_2_up = tf.nn.relu(conv2d(deconv1_end1_RGB_res, 64, 256, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_2_up"))
      de_conv_2_up2 = conv2d(de_conv_2_up, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_2_up2")
      de_conv_2_up3 = tf.nn.depth_to_space(de_conv_2_up2, block_size=2) # 通道数256/4=64
      # 残差连接第一层，融合特征 channel 64 concat 32  -> 96 -> 32
      # de_conv_2_tocat = tf.concat(axis = 3, values = [conv1_end1_RGB_res, conv1_end1_HSV_res, conv1_end1_LAB_res]) # axis=3表示第四个维度进行拼接
      de_conv_2_cat = tf.concat(axis = 3, values = [de_conv_2_up3, conv1_end1_RGB_res]) # axis=3表示第四个维度进行拼接
      toconcat_end_SCA_2 = self.Channel_Space_Attention_layer(de_conv_2_cat, out_dim=96, ratio=1, layer_name="toconcat_end_SCA_2")

    
      # 结束conv 得到 conv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=256)
      deconv2_end2_RGB_toRefine = tf.nn.relu(conv2d(toconcat_end_SCA_2, 96, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_RGB_toRefine"))
      deconv2_end2_RGB_refine1 = tf.nn.relu(conv2d(deconv2_end2_RGB_toRefine, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_RGB_refine1"))
      deconv2_end2_RGB_refine2 = tf.nn.relu(conv2d(deconv2_end2_RGB_refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_RGB_refine2"))
      deconv2_end2_RGB_res0 = conv2d(deconv2_end2_RGB_refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_RGB_res0")
      deconv2_end2_RGB_res1 = tf.add(deconv2_end2_RGB_res0, deconv2_end2_RGB_toRefine)
      deconv2_end2_RGB_toRefine1 = tf.nn.relu(conv2d(deconv2_end2_RGB_res1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_RGB_toRefine1"))
      deconv2_end2_RGB_Refine3 = tf.nn.relu(conv2d(deconv2_end2_RGB_toRefine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_RGB_Refine3"))
      deconv2_end2_RGB_Refine4 = tf.nn.relu(conv2d(deconv2_end2_RGB_Refine3, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_RGB_Refine4"))
      deconv2_end2_RGB_res2 = conv2d(deconv2_end2_RGB_Refine4, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_RGB_res2") 
      deconv2_end2_RGB_res = tf.add(deconv2_end2_RGB_res2,deconv2_end2_RGB_toRefine1) 
         
         
# Third RGB decoder
      # 亚像素卷积上采样
      de_conv_3_up = tf.nn.relu(conv2d(deconv2_end2_RGB_res, 32, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_3_up"))
      de_conv_3_up2 = conv2d(de_conv_3_up, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="de_conv_3_up2")
      de_conv_3_up3 = tf.nn.depth_to_space(de_conv_3_up2, block_size=2) # 通道数128/4=32
      # 残差连接第0层，融合特征 channel 32 concat 16  -> 48 -> 16
      # de_conv_3_tocat = tf.concat(axis = 3, values = [conv1_RGB, conv1_HSV, conv1_LAB]) # axis=3表示第四个维度进行拼接
      de_conv_3_cat = tf.concat(axis = 3, values = [de_conv_3_up3, conv1_RGB]) # axis=3表示第四个维度进行拼接
      toconcat_end_SCA_3 = self.Channel_Space_Attention_layer(de_conv_3_cat, out_dim=48, ratio=1, layer_name="toconcat_end_SCA_3")
      
      # 结束conv 得到 conv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=256)
      deconv3_end3_RGB_toadd = tf.nn.relu(conv2d(toconcat_end_SCA_3, 48, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_RGB_toadd"))
      # deconv3_end3_RGB_toRefine = tf.add(deconv3_end3_RGB_toadd, conv1_RGB_edge0) # 加入边缘信息
      deconv3_end3_RGB_refine1 = tf.nn.relu(conv2d(deconv3_end3_RGB_toadd, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_RGB_refine1"))
      deconv3_end3_RGB_refine2 = tf.nn.relu(conv2d(deconv3_end3_RGB_refine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_RGB_refine2"))
      deconv3_end3_RGB_res0 = conv2d(deconv3_end3_RGB_refine2, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_RGB_res0")
      deconv3_end3_RGB_res1 = tf.add(deconv3_end3_RGB_res0, deconv3_end3_RGB_toadd)
      deconv3_end3_RGB_toRefine1 = tf.nn.relu(conv2d(deconv3_end3_RGB_res1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_RGB_toRefine1"))
      deconv3_end3_RGB_Refine3 = tf.nn.relu(conv2d(deconv3_end3_RGB_toRefine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_RGB_Refine3"))
      deconv3_end3_RGB_Refine4 = tf.nn.relu(conv2d(deconv3_end3_RGB_Refine3, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_RGB_Refine4"))
      deconv3_end3_RGB_res2 = conv2d(deconv3_end3_RGB_Refine4, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_RGB_res2") 
      deconv3_end3_RGB_res = tf.add(deconv3_end3_RGB_res2,deconv3_end3_RGB_toRefine1) 
      
      # 得到最终结果
      final_results_RGB = tf.nn.sigmoid(conv2d(deconv3_end3_RGB_res, 16, 3, k_h=3, k_w=3, d_h=1, d_w=1,name="final_results_RGB"))    


# first noise encoder：输出 conv1_end1_RGB 3-64-128
     
      CM_cat_noise = tf.concat(axis = 3, values = [self.images_wb-self.images, self.images_gc-self.images, self.images_histeq-self.images]) # axis=3表示第四个维度（通道）进行拼接
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅
      conv1_noise_0 = tf.nn.tanh(conv2d(CM_cat_noise, 9, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noise_0"))
      conv1_noise_0_Refine1 = tf.nn.tanh(conv2d(conv1_noise_0, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noise_0_Refine1")) 
      conv1_noise_0_Refine2 = tf.nn.tanh(conv2d(conv1_noise_0_Refine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noise_0_Refine2")) 
      conv1_noise_res0 = conv2d(conv1_noise_0_Refine2, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noise_res0") 
      conv1_noise = tf.add(conv1_noise_res0,conv1_noise_0)# 与解码器残差连接
            
      # 多尺度卷积：64*H*W 变为 128*H/2*W/2，拼接后变为 noise_mt1 384*H/2*W/2
      # 1*1
      ### conv1_mt1_1_noise = tf.nn.tanh(conv2d(conv1_noise, 128,256,k_h=1, k_w=1, d_h=2, d_w=2,name="conv1_mt1_1_noise"))
      conv1_mt1_1_noise = tf.nn.tanh(conv2d(conv1_noise, 16, 32, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_noise"))
      # 3*3
      conv1_mt1_2_noise = tf.nn.tanh(conv2d(conv1_noise, 16, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_noise"))
      # 5*5
      conv1_mt1_3_noise = tf.nn.tanh(conv2d(conv1_noise, 16, 32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_noise"))
      
      noise_mt1= tf.concat(axis = 3, values = [conv1_mt1_1_noise,conv1_mt1_2_noise,conv1_mt1_3_noise]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=128*3=384)
      noise_mt1_SCA = self.Channel_Space_Attention_layer(noise_mt1, out_dim=96, ratio=1, layer_name="noise_mt1_SCA")

      # 多尺度卷积
      
     
      # 结束conv 得到 conv1_end1_noise：(batch_size, height=H/2, width=W/2, channels=128)
      conv1_end1_noise_toRefine = tf.nn.tanh(conv2d(noise_mt1_SCA, 96, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_noise_toRefine"))
      conv1_end1_noise_Refine1 = tf.nn.tanh(conv2d(conv1_end1_noise_toRefine, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_noise_Refine1"))
      conv1_end1_noise_Refine2 = tf.nn.tanh(conv2d(conv1_end1_noise_Refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_noise_Refine2"))
      conv1_end1_noise_res0 = conv2d(conv1_end1_noise_Refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_noise_res0") 
      conv1_end1_noise_res1 = tf.add(conv1_end1_noise_toRefine, conv1_end1_noise_res0) # 与解码器特征融合
      conv1_end1_noise_res = tf.nn.max_pool(conv1_end1_noise_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Second noise encoder：输出 conv2_end2_noise 128-256
      noise_input_scale2 =  tf.nn.max_pool(CM_cat_noise, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      conv1_noisescale2_0 = tf.nn.relu(conv2d(noise_input_scale2, 9, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noisescale2_0"))
      conv1_noisescale2_0_Refine1 = tf.nn.relu(conv2d(conv1_noisescale2_0, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noisescale2_0_Refine1")) 
      conv1_noisescale2_0_Refine2 = tf.nn.relu(conv2d(conv1_noisescale2_0_Refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noisescale2_0_Refine2")) 
      conv1_noisescale2_res0 = conv2d(conv1_noisescale2_0_Refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noisescale2_res0") 
      conv1_noisescale2 = tf.add(conv1_noisescale2_res0,conv1_noisescale2_0)# 与解码器残差连接
      noise_input2 = tf.concat(axis = 3, values = [conv1_end1_noise_res, conv1_noisescale2])
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅      
      # 多尺度卷积：128*H/2*W/2 变为 256*H/2*W/2，拼接后变为 noise_mt2 768*H/2*W/2
      # 1*1
      conv2_mt2_1_noise = tf.nn.tanh(conv2d(noise_input2, 64, 64, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_noise"))
      # 3*3
      conv2_mt2_2_noise = tf.nn.tanh(conv2d(noise_input2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_noise"))
      # 5*5
      conv2_mt2_3_noise = tf.nn.tanh(conv2d(noise_input2, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_noise"))
      
      noise_mt2= tf.concat(axis = 3, values = [conv2_mt2_1_noise, conv2_mt2_2_noise, conv2_mt2_3_noise]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=256*3=768)
      noise_mt2_SCA = self.Channel_Space_Attention_layer(noise_mt2, out_dim=192, ratio=1, layer_name="noise_mt2_SCA")

      # 多尺度卷积
      
      # 结束conv 得到 conv1_end1_noise：(batch_size, height=H/2, width=W/2, channels=256)
      conv2_end2_noise_toRefine = tf.nn.tanh(conv2d(noise_mt2_SCA, 192, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_noise_toRefine"))
      conv2_end2_noise_refine1 = tf.nn.tanh(conv2d(conv2_end2_noise_toRefine, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_noise_refine1"))
      conv2_end2_noise_refine2 = tf.nn.tanh(conv2d(conv2_end2_noise_refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_noise_refine2"))
      conv2_end2_noise_res0 = conv2d(conv2_end2_noise_refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_noise_res0")
      conv2_end2_noise_res1 = tf.add(conv2_end2_noise_toRefine, conv2_end2_noise_res0)
      conv2_end2_noise_res = tf.nn.max_pool(conv2_end2_noise_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

         
# Third noise encoder：
      noise_input_scale3 =  tf.nn.max_pool(CM_cat_noise, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
      conv1_noisescale3_0 = tf.nn.relu(conv2d(noise_input_scale3, 9, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noisescale3_0"))
      conv1_noisescale3_0_Refine1 = tf.nn.relu(conv2d(conv1_noisescale3_0, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noisescale3_0_Refine1")) 
      conv1_noisescale3_0_Refine2 = tf.nn.relu(conv2d(conv1_noisescale3_0_Refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noisescale3_0_Refine2")) 
      conv1_noisescale3_res0 = conv2d(conv1_noisescale3_0_Refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_noisescale3_res0") 
      conv1_noisescale3 = tf.add(conv1_noisescale3_res0,conv1_noisescale3_0)# 与解码器残差连接
      noise_input3 = tf.concat(axis = 3, values = [conv2_end2_noise_res, conv1_noisescale3])
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅      
      # 多尺度卷积：
      # 1*1
      conv3_mt3_1_noise = tf.nn.tanh(conv2d(noise_input3, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_noise"))
      # 3*3
      conv3_mt3_2_noise = tf.nn.tanh(conv2d(noise_input3, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_noise"))
      # 5*5
      conv3_mt3_3_noise = tf.nn.tanh(conv2d(noise_input3, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_noise"))
      
      noise_mt3 = tf.concat(axis = 3, values = [conv3_mt3_1_noise, conv3_mt3_2_noise, conv3_mt3_3_noise]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=256*3=768)
      noise_mt3_SCA = self.Channel_Space_Attention_layer(noise_mt3, out_dim=384, ratio=1, layer_name="noise_mt3_SCA")
      # 多尺度卷积
      
      # 结束conv 得到 conv1_end1_noise：(batch_size, height=H/2, width=W/2, channels=256)
      conv3_end3_noise_toRefine = tf.nn.tanh(conv2d(noise_mt3_SCA, 384, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_noise_toRefine"))
      conv3_end3_noise_refine1 = tf.nn.tanh(conv2d(conv3_end3_noise_toRefine, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_noise_refine1"))
      conv3_end3_noise_refine2 = tf.nn.tanh(conv2d(conv3_end3_noise_refine1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_noise_refine2"))
      conv3_end3_noise_res0 = conv2d(conv3_end3_noise_refine2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_noise_res0")
      conv3_end3_noise_res1 = tf.add(conv3_end3_noise_toRefine, conv3_end3_noise_res0)
      conv3_end3_noise_res = tf.nn.max_pool(conv3_end3_noise_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
         
      fusion_res_SCA_noise = self.Channel_Space_Attention_layer(conv3_end3_noise_res, out_dim=128, ratio=1, layer_name="fusion_res_SCA_noise")
 
  # Zero decoder
    # pre 精细化 512
      denoise_conv_0_re0 = tf.nn.tanh(conv2d(fusion_res_SCA_noise, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_0_re0"))
      denoise_conv_0_re1 = tf.nn.tanh(conv2d(denoise_conv_0_re0, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_0_re1"))
      denoise_conv_0_re2 = tf.nn.tanh(conv2d(denoise_conv_0_re1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_0_re2"))
      denoise_conv_0_re3 = conv2d(denoise_conv_0_re2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_0_re3")
      denoise_conv_0_re1_add = tf.add(denoise_conv_0_re3, denoise_conv_0_re0)
      
      # 获取置信度图 Confidence Map
      CM_Noise_cat = tf.concat(axis = 3, values = [self.images_wb-self.images, self.images_gc-self.images, self.images_histeq-self.images]) # axis=3表示第四个维度（通道）进行拼接
      CM_Noise1_pool = tf.nn.max_pool(CM_Noise_cat, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
      CM_Noise1_conv1 = tf.nn.tanh(conv2d(CM_Noise1_pool, 9, 128, k_h=7, k_w=7, d_h=1, d_w=1,name="CM_Noise1_conv1"))
      CM_Noise1_conv2 = tf.nn.tanh(conv2d(CM_Noise1_conv1, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="CM_Noise1_conv2"))
      CM_Noise1_conv3 = tf.nn.tanh(conv2d(CM_Noise1_conv2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="CM_Noise1_conv3"))
      CM_Noise1_conv4 = tf.nn.tanh(conv2d(CM_Noise1_conv3, 128, 64, k_h=1, k_w=1, d_h=1, d_w=1,name="CM_Noise1_conv4"))
      CM_Noise2_conv1 = tf.nn.tanh(conv2d(CM_Noise1_conv4, 64, 64, k_h=7, k_w=7, d_h=1, d_w=1,name="CM_Noise2_conv1"))
      CM_Noise2_conv2 = tf.nn.tanh(conv2d(CM_Noise2_conv1, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1,name="CM_Noise2_conv2"))
      CM_Noise2_conv3 = tf.nn.tanh(conv2d(CM_Noise2_conv2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="CM_Noise2_conv3"))
      CM_Noise2_conv4 = tf.nn.sigmoid(conv2d(CM_Noise2_conv3, 64, 3, k_h=3, k_w=3, d_h=1, d_w=1,name="CM_Noise2_conv4"))
      out_wb_noise, out_gc_noise, out_histeq_noise = tf.split(CM_Noise2_conv4, num_or_size_splits=[1, 1, 1], axis=-1)
      
      # 结果乘以置信度图获得最终结果
      denoise_conv_0_re1_add1 = tf.multiply(denoise_conv_0_re1_add, out_wb_noise) + tf.multiply(denoise_conv_0_re1_add, out_gc_noise) + tf.multiply(denoise_conv_0_re1_add, out_histeq_noise)
      

# first decoder
      # 亚像素卷积上采样
      denoise_conv_1_up = tf.nn.tanh(conv2d(denoise_conv_0_re1_add1, 128, 512, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_1_up"))
      denoise_conv_1_up2 = conv2d(denoise_conv_1_up, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_1_up2")
      denoise_conv_1_up3 = tf.nn.depth_to_space(denoise_conv_1_up2, block_size=2) # 通道数512/4=128
      # 残差连接第二层，融合特征 channel 128 concat 64  -> 192 -> 64
      # de_conv_1_tocat = tf.concat(axis = 3, values = [conv2_end2_noise_res, conv2_end2_HSV_res, conv2_end2_LAB_res]) # axis=3表示第四个维度进行拼接
      denoise_conv_1_cat = tf.concat(axis = 3, values = [denoise_conv_1_up3, conv2_end2_noise_res]) # axis=3表示第四个维度进行拼接
      toconcat_end_SCA_1_noise = self.Channel_Space_Attention_layer(denoise_conv_1_cat, out_dim=192, ratio=1, layer_name="toconcat_end_SCA_1_noise")

      
      # 结束conv 得到 deconv1_end1_noise：(batch_size, height=H/2, width=W/2, channels=128)
      deconv1_end1_noise_toRefine = tf.nn.tanh(conv2d(toconcat_end_SCA_1_noise, 192, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_noise_toRefine"))
      deconv1_end1_noise_Refine1 = tf.nn.tanh(conv2d(deconv1_end1_noise_toRefine, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_noise_Refine1"))
      deconv1_end1_noise_Refine2 = tf.nn.tanh(conv2d(deconv1_end1_noise_Refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_noise_Refine2"))
      deconv1_end1_noise_res0 = conv2d(deconv1_end1_noise_Refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_noise_res0") 
      deconv1_end1_noise_res1 = tf.add(deconv1_end1_noise_res0,deconv1_end1_noise_toRefine) 
      deconv1_end1_noise_toRefine1 = tf.nn.tanh(conv2d(deconv1_end1_noise_res1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_noise_toRefine1"))
      deconv1_end1_noise_Refine3 = tf.nn.tanh(conv2d(deconv1_end1_noise_toRefine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_noise_Refine3"))
      deconv1_end1_noise_Refine4 = tf.nn.tanh(conv2d(deconv1_end1_noise_Refine3, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_noise_Refine4"))
      deconv1_end1_noise_res2 = conv2d(deconv1_end1_noise_Refine4, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv1_end1_noise_res2") 
      deconv1_end1_noise_res = tf.add(deconv1_end1_noise_res2,deconv1_end1_noise_toRefine1) 
      


# Second decoder
      # 亚像素卷积上采样
      denoise_conv_2_up = tf.nn.tanh(conv2d(deconv1_end1_noise_res, 64, 256, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_2_up"))
      denoise_conv_2_up2 = conv2d(denoise_conv_2_up, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_2_up2")
      denoise_conv_2_up3 = tf.nn.depth_to_space(denoise_conv_2_up2, block_size=2) # 通道数256/4=64
      # 残差连接第一层，融合特征 channel 64 concat 32  -> 96 -> 32
      # de_conv_2_tocat = tf.concat(axis = 3, values = [conv1_end1_noise_res, conv1_end1_HSV_res, conv1_end1_LAB_res]) # axis=3表示第四个维度进行拼接
      denoise_conv_2_cat = tf.concat(axis = 3, values = [denoise_conv_2_up3, conv1_end1_noise_res]) # axis=3表示第四个维度进行拼接
      toconcat_end_SCA_2_noise = self.Channel_Space_Attention_layer(denoise_conv_2_cat, out_dim=96, ratio=1, layer_name="toconcat_end_SCA_2_noise")

    
      # 结束conv 得到 conv1_end1_noise：(batch_size, height=H/2, width=W/2, channels=256)
      deconv2_end2_noise_toRefine = tf.nn.tanh(conv2d(toconcat_end_SCA_2_noise, 96, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_noise_toRefine"))
      deconv2_end2_noise_refine1 = tf.nn.tanh(conv2d(deconv2_end2_noise_toRefine, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_noise_refine1"))
      deconv2_end2_noise_refine2 = tf.nn.tanh(conv2d(deconv2_end2_noise_refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_noise_refine2"))
      deconv2_end2_noise_res0 = conv2d(deconv2_end2_noise_refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_noise_res0")
      deconv2_end2_noise_res1 = tf.add(deconv2_end2_noise_res0, deconv2_end2_noise_toRefine)
      deconv2_end2_noise_toRefine1 = tf.nn.tanh(conv2d(deconv2_end2_noise_res1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_noise_toRefine1"))
      deconv2_end2_noise_Refine3 = tf.nn.tanh(conv2d(deconv2_end2_noise_toRefine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_noise_Refine3"))
      deconv2_end2_noise_Refine4 = tf.nn.tanh(conv2d(deconv2_end2_noise_Refine3, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_noise_Refine4"))
      deconv2_end2_noise_res2 = conv2d(deconv2_end2_noise_Refine4, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv2_end2_noise_res2") 
      deconv2_end2_noise_res = tf.add(deconv2_end2_noise_res2,deconv2_end2_noise_toRefine1) 
         
         
# Third noise decoder
      # 亚像素卷积上采样
      denoise_conv_3_up = tf.nn.tanh(conv2d(deconv2_end2_noise_res, 32, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_3_up"))
      denoise_conv_3_up2 = conv2d(denoise_conv_3_up, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="denoise_conv_3_up2")
      denoise_conv_3_up3 = tf.nn.depth_to_space(denoise_conv_3_up2, block_size=2) # 通道数128/4=32
      # 残差连接第0层，融合特征 channel 32 concat 16  -> 48 -> 16
      # de_conv_3_tocat = tf.concat(axis = 3, values = [conv1_noise, conv1_HSV, conv1_LAB]) # axis=3表示第四个维度进行拼接
      denoise_conv_3_cat = tf.concat(axis = 3, values = [denoise_conv_3_up3, conv1_noise]) # axis=3表示第四个维度进行拼接
      toconcat_end_SCA_3_noise = self.Channel_Space_Attention_layer(denoise_conv_3_cat, out_dim=48, ratio=1, layer_name="toconcat_end_SCA_3_noise")
      
      # 结束conv 得到 conv1_end1_noise：(batch_size, height=H/2, width=W/2, channels=256)
      deconv3_end3_noise_toadd = tf.nn.tanh(conv2d(toconcat_end_SCA_3_noise, 48, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_noise_toadd"))
      # deconv3_end3_noise_toRefine = tf.add(deconv3_end3_noise_toadd, conv1_noise_edge0) # 加入边缘信息
      deconv3_end3_noise_refine1 = tf.nn.tanh(conv2d(deconv3_end3_noise_toadd, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_noise_refine1"))
      deconv3_end3_noise_refine2 = tf.nn.tanh(conv2d(deconv3_end3_noise_refine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_noise_refine2"))
      deconv3_end3_noise_res0 = conv2d(deconv3_end3_noise_refine2, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_noise_res0")
      deconv3_end3_noise_res1 = tf.add(deconv3_end3_noise_res0, deconv3_end3_noise_toadd)
      deconv3_end3_noise_toRefine1 = tf.nn.tanh(conv2d(deconv3_end3_noise_res1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_noise_toRefine1"))
      deconv3_end3_noise_Refine3 = tf.nn.tanh(conv2d(deconv3_end3_noise_toRefine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_noise_Refine3"))
      deconv3_end3_noise_Refine4 = tf.nn.tanh(conv2d(deconv3_end3_noise_Refine3, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_noise_Refine4"))
      deconv3_end3_noise_res2 = conv2d(deconv3_end3_noise_Refine4, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deconv3_end3_noise_res2") 
      deconv3_end3_noise_res = tf.add(deconv3_end3_noise_res2,deconv3_end3_noise_toRefine1) 
      
      # 得到最终结果
      final_results_noise = tf.nn.tanh(conv2d(deconv3_end3_noise_res, 16, 3, k_h=3, k_w=3, d_h=1, d_w=1,name="final_results_noise"))    



# first LAB encoder：
      LAB1 = Conv_img.rgb_to_lab(self.images)
      LAB2 = Conv_img.rgb_to_lab(self.images_wb)
      LAB3 = Conv_img.rgb_to_lab(self.images_gc)
      LAB4 = Conv_img.rgb_to_lab(self.images_histeq)
      CM_cat_LAB = tf.concat(axis = 3, values = [LAB1, LAB2, LAB3, LAB4]) # axis=3表示第四个维度（通道）进行拼接
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅
      conv1_LAB_0 = tf.nn.relu(conv2d(CM_cat_LAB, 12, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LAB_0"))
      conv1_LAB_0_Refine1 = tf.nn.relu(conv2d(conv1_LAB_0, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LAB_0_Refine1")) 
      conv1_LAB_0_Refine2 = tf.nn.relu(conv2d(conv1_LAB_0_Refine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LAB_0_Refine2")) 
      conv1_LAB_res0 = conv2d(conv1_LAB_0_Refine2, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LAB_res0") 
      conv1_LAB = tf.add(conv1_LAB_res0,conv1_LAB_0)# 与解码器残差连接
      
      # 多尺度卷积：64*H*W 变为 128*H/2*W/2，拼接后变为 LAB_mt1 384*H/2*W/2
      # 1*1
      ### conv1_mt1_1_LAB = tf.nn.relu(conv2d(conv1_LAB, 128,256,k_h=1, k_w=1, d_h=2, d_w=2,name="conv1_mt1_1_LAB"))
      conv1_mt1_1_LAB = tf.nn.relu(conv2d(conv1_LAB, 16, 32, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_LAB"))
      # conv1_mt1_1_1_LAB = tf.nn.relu(conv2d(conv1_mt1_1_LAB, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_1_LAB"))
      # conv1_mt1_1_2_LAB = tf.nn.relu(conv2d(conv1_mt1_1_1_LAB, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_2_LAB"))
      # conv1_mt1_1_3_LAB = conv2d(conv1_mt1_1_2_LAB, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1,name="conv1_mt1_1_3_LAB")
      # conv1_mt1_1_4_LAB = tf.add(conv1_mt1_1_LAB, conv1_mt1_1_3_LAB)
      # 3*3
      conv1_mt1_2_LAB = tf.nn.relu(conv2d(conv1_LAB, 16, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_LAB"))
      # conv1_mt1_2_1_LAB = tf.nn.relu(conv2d(conv1_mt1_2_LAB, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_1_LAB"))
      # conv1_mt1_2_2_LAB = tf.nn.relu(conv2d(conv1_mt1_2_1_LAB, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_2_LAB"))
      # conv1_mt1_2_3_LAB = conv2d(conv1_mt1_2_2_LAB, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_mt1_2_3_LAB")
      # conv1_mt1_2_4_LAB = tf.add(conv1_mt1_2_LAB, conv1_mt1_2_3_LAB)
      # 5*5
      conv1_mt1_3_LAB = tf.nn.relu(conv2d(conv1_LAB, 16, 32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_LAB"))
      # conv1_mt1_3_1_LAB = tf.nn.relu(conv2d(conv1_mt1_3_LAB, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_1_LAB"))
      # conv1_mt1_3_2_LAB = tf.nn.relu(conv2d(conv1_mt1_3_1_LAB, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_2_LAB"))
      # conv1_mt1_3_3_LAB = conv2d(conv1_mt1_3_2_LAB, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="conv1_mt1_3_3_LAB")
      # conv1_mt1_3_4_LAB = tf.add(conv1_mt1_3_LAB, conv1_mt1_3_3_LAB)
      
      LAB_mt1= tf.concat(axis = 3, values = [conv1_mt1_1_LAB,conv1_mt1_2_LAB,conv1_mt1_3_LAB]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=128*3=384)
      LAB_mt1_SCA = self.Channel_Space_Attention_layer(LAB_mt1, out_dim=96, ratio=1, layer_name="LAB_mt1_SCA")
      # 多尺度卷积
     
      # 结束conv 得到 conv1_end1_LAB：(batch_size, height=H/2, width=W/2, channels=128)
      conv1_end1_LAB_toRefine = tf.nn.relu(conv2d(LAB_mt1_SCA, 96, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_LAB_toRefine"))
      conv1_end1_LAB_Refine1 = tf.nn.relu(conv2d(conv1_end1_LAB_toRefine, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_LAB_Refine1"))
      conv1_end1_LAB_Refine2 = tf.nn.relu(conv2d(conv1_end1_LAB_Refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_LAB_Refine2"))
      conv1_end1_LAB_res0 = conv2d(conv1_end1_LAB_Refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_end1_LAB_res0") 
      conv1_end1_LAB_res1 = tf.add(conv1_end1_LAB_toRefine, conv1_end1_LAB_res0) # 与解码器特征融合
      conv1_end1_LAB_res = tf.nn.max_pool(conv1_end1_LAB_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Second LAB encoder：输出 conv2_end2_LAB 128-256
      LAB_input_scale2 =  tf.nn.max_pool(CM_cat_LAB, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      conv1_LABscale2_0 = tf.nn.relu(conv2d(LAB_input_scale2, 12, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LABscale2_0"))
      conv1_LABscale2_0_Refine1 = tf.nn.relu(conv2d(conv1_LABscale2_0, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LABscale2_0_Refine1")) 
      conv1_LABscale2_0_Refine2 = tf.nn.relu(conv2d(conv1_LABscale2_0_Refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LABscale2_0_Refine2")) 
      conv1_LABscale2_res0 = conv2d(conv1_LABscale2_0_Refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LABscale2_res0") 
      conv1_LABscale2 = tf.add(conv1_LABscale2_res0,conv1_LABscale2_0)# 与解码器残差连接
      LAB_input2 = tf.concat(axis = 3, values = [conv1_end1_LAB_res, conv1_LABscale2])
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅      
      # 多尺度卷积：128*H/2*W/2 变为 256*H/2*W/2，拼接后变为 LAB_mt2 768*H/2*W/2
      # 1*1
      conv2_mt2_1_LAB = tf.nn.relu(conv2d(LAB_input2, 64, 64, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_LAB"))
      # conv2_mt2_1_1_LAB = tf.nn.relu(conv2d(conv2_mt2_1_LAB, 256, 256, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_1_LAB"))
      # conv2_mt2_1_2_LAB = tf.nn.relu(conv2d(conv2_mt2_1_1_LAB, 256, 256, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_2_LAB"))
      # conv2_mt2_1_3_LAB = tf.nn.relu(conv2d(conv2_mt2_1_2_LAB, 256, 256, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2_mt2_1_3_LAB"))
      # conv2_mt2_1_4_LAB = tf.add(conv2_mt2_1_LAB, conv2_mt2_1_3_LAB)
      # 3*3
      conv2_mt2_2_LAB = tf.nn.relu(conv2d(LAB_input2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_LAB"))
      # conv2_mt2_2_1_LAB = tf.nn.relu(conv2d(conv2_mt2_2_LAB, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_1_LAB"))
      # conv2_mt2_2_2_LAB = tf.nn.relu(conv2d(conv2_mt2_2_1_LAB, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_2_LAB"))
      # conv2_mt2_2_3_LAB = conv2d(conv2_mt2_2_2_LAB, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2_mt2_2_3_LAB")
      # conv2_mt2_2_4_LAB = tf.add(conv2_mt2_2_LAB, conv2_mt2_2_3_LAB)
      # 5*5
      conv2_mt2_3_LAB = tf.nn.relu(conv2d(LAB_input2, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_LAB"))
      # conv2_mt2_3_1_LAB = tf.nn.relu(conv2d(conv2_mt2_3_LAB, 256, 256, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_1_LAB"))
      # conv2_mt2_3_2_LAB = tf.nn.relu(conv2d(conv2_mt2_3_1_LAB, 256, 256, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_2_LAB"))
      # conv2_mt2_3_3_LAB = conv2d(conv2_mt2_3_2_LAB, 256, 256, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2_mt2_3_3_LAB")
      # conv2_mt2_3_4_LAB = tf.add(conv2_mt2_3_LAB, conv2_mt2_3_3_LAB)
      
      LAB_mt2= tf.concat(axis = 3, values = [conv2_mt2_1_LAB, conv2_mt2_2_LAB, conv2_mt2_3_LAB]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=256*3=768)
      LAB_mt2_SCA = self.Channel_Space_Attention_layer(LAB_mt2, out_dim=192, ratio=1, layer_name="LAB_mt2_SCA")
      # 多尺度卷积
      
      # 结束conv 得到 conv1_end1_LAB：(batch_size, height=H/2, width=W/2, channels=256)
      conv2_end2_LAB_toRefine = tf.nn.relu(conv2d(LAB_mt2_SCA, 192, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_LAB_toRefine"))
      conv2_end2_LAB_refine1 = tf.nn.relu(conv2d(conv2_end2_LAB_toRefine, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_LAB_refine1"))
      conv2_end2_LAB_refine2 = tf.nn.relu(conv2d(conv2_end2_LAB_refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_LAB_refine2"))
      conv2_end2_LAB_res0 = conv2d(conv2_end2_LAB_refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2_end2_LAB_res0")
      conv2_end2_LAB_res1 = tf.add(conv2_end2_LAB_toRefine, conv2_end2_LAB_res0)
      conv2_end2_LAB_res = tf.nn.max_pool(conv2_end2_LAB_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

         
# Third LAB encoder：
      LAB_input_scale3 =  tf.nn.max_pool(CM_cat_LAB, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
      conv1_LABscale3_0 = tf.nn.relu(conv2d(LAB_input_scale3, 12, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LABscale3_0"))
      conv1_LABscale3_0_Refine1 = tf.nn.relu(conv2d(conv1_LABscale3_0, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LABscale3_0_Refine1")) 
      conv1_LABscale3_0_Refine2 = tf.nn.relu(conv2d(conv1_LABscale3_0_Refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LABscale3_0_Refine2")) 
      conv1_LABscale3_res0 = conv2d(conv1_LABscale3_0_Refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv1_LABscale3_res0") 
      conv1_LABscale3 = tf.add(conv1_LABscale3_res0,conv1_LABscale3_0)# 与解码器残差连接
      LAB_input3 = tf.concat(axis = 3, values = [conv2_end2_LAB_res, conv1_LABscale3])
      # 四维输入数据，输入通道数，输出通道数/卷积核数量，卷积核高，卷积核宽，水平步幅，垂直步幅      
      # 多尺度卷积：
      # 1*1
      conv3_mt3_1_LAB = tf.nn.relu(conv2d(LAB_input3, 128, 128, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_LAB"))
      # conv3_mt3_1_1_LAB = tf.nn.relu(conv2d(conv3_mt3_1_LAB, 512, 512, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_1_LAB"))
      # conv3_mt3_1_2_LAB = tf.nn.relu(conv2d(conv3_mt3_1_1_LAB, 512, 512, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_2_LAB"))
      # conv3_mt3_1_3_LAB = conv2d(conv3_mt3_1_2_LAB, 512, 512, k_h=1, k_w=1, d_h=1, d_w=1, name="conv3_mt3_1_3_LAB")
      # conv3_mt3_1_4_LAB = tf.add(conv3_mt3_1_LAB, conv3_mt3_1_3_LAB)
      # 3*3
      conv3_mt3_2_LAB = tf.nn.relu(conv2d(LAB_input3, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_LAB"))
      # conv3_mt3_2_1_LAB = tf.nn.relu(conv2d(conv3_mt3_2_LAB, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_1_LAB"))
      # conv3_mt3_2_2_LAB = tf.nn.relu(conv2d(conv3_mt3_2_1_LAB, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_2_LAB"))
      # conv3_mt3_2_3_LAB = conv2d(conv3_mt3_2_2_LAB, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1, name="conv3_mt3_2_3_LAB")
      # conv3_mt3_2_4_LAB = tf.add(conv3_mt3_2_LAB, conv3_mt3_2_3_LAB)
      # 5*5
      conv3_mt3_3_LAB = tf.nn.relu(conv2d(LAB_input3, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_LAB"))
      # conv3_mt3_3_1_LAB = tf.nn.relu(conv2d(conv3_mt3_3_LAB, 512, 512, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_1_LAB"))
      # conv3_mt3_3_2_LAB = tf.nn.relu(conv2d(conv3_mt3_3_1_LAB, 512, 512, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_2_LAB"))
      # conv3_mt3_3_3_LAB = conv2d(conv3_mt3_3_2_LAB, 512, 512, k_h=5, k_w=5, d_h=1, d_w=1, name="conv3_mt3_3_3_LAB")
      # conv3_mt3_3_4_LAB = tf.add(conv3_mt3_3_LAB, conv3_mt3_3_3_LAB)
      
      LAB_mt3 = tf.concat(axis = 3, values = [conv3_mt3_1_LAB, conv3_mt3_2_LAB, conv3_mt3_3_LAB]) # axis=3表示第四个维度进行拼接，形状为 (batch_size, height=H/2, width=W/2, channels=256*3=768)
      LAB_mt3_SCA = self.Channel_Space_Attention_layer(LAB_mt3, out_dim=384, ratio=1, layer_name="LAB_mt3_SCA")
      # 多尺度卷积
      
      # 结束conv 得到 conv1_end1_LAB：(batch_size, height=H/2, width=W/2, channels=256)
      conv3_end3_LAB_toRefine = tf.nn.relu(conv2d(LAB_mt3_SCA, 384, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_LAB_toRefine"))
      conv3_end3_LAB_refine1 = tf.nn.relu(conv2d(conv3_end3_LAB_toRefine, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_LAB_refine1"))
      conv3_end3_LAB_refine2 = tf.nn.relu(conv2d(conv3_end3_LAB_refine1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_LAB_refine2"))
      conv3_end3_LAB_res0 = conv2d(conv3_end3_LAB_refine2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv3_end3_LAB_res0")
      conv3_end3_LAB_res1 = tf.add(conv3_end3_LAB_toRefine, conv3_end3_LAB_res0)
      conv3_end3_LAB_res = tf.nn.max_pool(conv3_end3_LAB_res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
         
      fusion_res_SCA_LAB = self.Channel_Space_Attention_layer(conv3_end3_LAB_res, out_dim=128, ratio=1, layer_name="fusion_res_SCA_LAB")
    
    
    
  # Zero decoder
    # pre 精细化 512
      deLAB_conv_0_re0 = tf.nn.relu(conv2d(fusion_res_SCA_LAB, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_0_re0"))
      deLAB_conv_0_re1 = tf.nn.relu(conv2d(deLAB_conv_0_re0, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_0_re1"))
      deLAB_conv_0_re2 = tf.nn.relu(conv2d(deLAB_conv_0_re1, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_0_re2"))
      deLAB_conv_0_re3 = conv2d(deLAB_conv_0_re2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_0_re3")
      deLAB_conv_0_re1_add = tf.add(deLAB_conv_0_re3, deLAB_conv_0_re0)
      
      # 获取置信度图 Confidence Map
      CM_LAB_cat = tf.concat(axis = 3, values = [LAB1, LAB2, LAB3, LAB4]) # axis=3表示第四个维度（通道）进行拼接
      CM_LAB1_pool = tf.nn.max_pool(CM_LAB_cat, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
      CM_LAB1_conv1 = tf.nn.tanh(conv2d(CM_LAB1_pool, 12, 128, k_h=7, k_w=7, d_h=1, d_w=1,name="CM_LAB1_conv1"))
      CM_LAB1_conv2 = tf.nn.tanh(conv2d(CM_LAB1_conv1, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1,name="CM_LAB1_conv2"))
      CM_LAB1_conv3 = tf.nn.tanh(conv2d(CM_LAB1_conv2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="CM_LAB1_conv3"))
      CM_LAB1_conv4 = tf.nn.tanh(conv2d(CM_LAB1_conv3, 128, 64, k_h=1, k_w=1, d_h=1, d_w=1,name="CM_LAB1_conv4"))
      CM_LAB2_conv1 = tf.nn.tanh(conv2d(CM_LAB1_conv4, 64, 64, k_h=7, k_w=7, d_h=1, d_w=1,name="CM_LAB2_conv1"))
      CM_LAB2_conv2 = tf.nn.tanh(conv2d(CM_LAB2_conv1, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1,name="CM_LAB2_conv2"))
      CM_LAB2_conv3 = tf.nn.tanh(conv2d(CM_LAB2_conv2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="CM_LAB2_conv3"))
      CM_LAB2_conv4 = tf.nn.sigmoid(conv2d(CM_LAB2_conv3, 64, 3, k_h=3, k_w=3, d_h=1, d_w=1,name="CM_LAB2_conv4"))
      out_wb_LAB, out_gc_LAB, out_histeq_LAB = tf.split(CM_LAB2_conv4, num_or_size_splits=[1, 1, 1], axis=-1)
      
      # 结果乘以置信度图获得最终结果
      deLAB_conv_0_re1_add1 = tf.multiply(deLAB_conv_0_re1_add, out_wb_LAB) + tf.multiply(deLAB_conv_0_re1_add, out_gc_LAB) + tf.multiply(deLAB_conv_0_re1_add, out_histeq_LAB)
      


# first decoder
      # 亚像素卷积上采样
      deLAB_conv_1_up = tf.nn.relu(conv2d(deLAB_conv_0_re1_add1, 128, 512, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_1_up"))
      deLAB_conv_1_up2 = conv2d(deLAB_conv_1_up, 512, 512, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_1_up2")
      deLAB_conv_1_up3 = tf.nn.depth_to_space(deLAB_conv_1_up2, block_size=2) # 通道数512/4=128
      # 残差连接第二层，融合特征 channel 128 concat 64  -> 192 -> 64
      # deLAB_conv_1_tocat = tf.concat(axis = 3, values = [conv2_end2_RGB_res, conv2_end2_HSV_res, conv2_end2_LAB_res]) # axis=3表示第四个维度进行拼接
      deLAB_conv_1_cat = tf.concat(axis = 3, values = [deLAB_conv_1_up3, conv2_end2_LAB_res]) # axis=3表示第四个维度进行拼接
      toconcat_end_SCA_1_LAB = self.Channel_Space_Attention_layer(deLAB_conv_1_cat, out_dim=192, ratio=1, layer_name="toconcat_end_SCA_1_LAB")

      
      # 结束conv 得到 deLABconv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=128)
      deLABconv1_end1_RGB_toRefine = tf.nn.relu(conv2d(toconcat_end_SCA_1_LAB, 192, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv1_end1_RGB_toRefine"))
      deLABconv1_end1_RGB_Refine1 = tf.nn.relu(conv2d(deLABconv1_end1_RGB_toRefine, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv1_end1_RGB_Refine1"))
      deLABconv1_end1_RGB_Refine2 = tf.nn.relu(conv2d(deLABconv1_end1_RGB_Refine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv1_end1_RGB_Refine2"))
      deLABconv1_end1_RGB_res0 = conv2d(deLABconv1_end1_RGB_Refine2, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv1_end1_RGB_res0") 
      deLABconv1_end1_RGB_res1 = tf.add(deLABconv1_end1_RGB_res0,deLABconv1_end1_RGB_toRefine) 
      deLABconv1_end1_RGB_toRefine1 = tf.nn.relu(conv2d(deLABconv1_end1_RGB_res1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv1_end1_RGB_toRefine1"))
      deLABconv1_end1_RGB_Refine3 = tf.nn.relu(conv2d(deLABconv1_end1_RGB_toRefine1, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv1_end1_RGB_Refine3"))
      deLABconv1_end1_RGB_Refine4 = tf.nn.relu(conv2d(deLABconv1_end1_RGB_Refine3, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv1_end1_RGB_Refine4"))
      deLABconv1_end1_RGB_res2 = conv2d(deLABconv1_end1_RGB_Refine4, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv1_end1_RGB_res2") 
      deLABconv1_end1_RGB_res = tf.add(deLABconv1_end1_RGB_res2,deLABconv1_end1_RGB_toRefine1) 
      


# Second decoder
      # 亚像素卷积上采样
      deLAB_conv_2_up = tf.nn.relu(conv2d(deLABconv1_end1_RGB_res, 64, 256, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_2_up"))
      deLAB_conv_2_up2 = conv2d(deLAB_conv_2_up, 256, 256, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_2_up2")
      deLAB_conv_2_up3 = tf.nn.depth_to_space(deLAB_conv_2_up2, block_size=2) # 通道数256/4=64
      # 残差连接第一层，融合特征 channel 64 concat 32  -> 160 -> 32
      # deLAB_conv_2_tocat = tf.concat(axis = 3, values = [conv1_end1_RGB_res, conv1_end1_HSV_res, conv1_end1_LAB_res]) # axis=3表示第四个维度进行拼接
      deLAB_conv_2_cat = tf.concat(axis = 3, values = [deLAB_conv_2_up3, conv1_end1_LAB_res]) # axis=3表示第四个维度进行拼接
      toconcat_end_SCA_2_LAB = self.Channel_Space_Attention_layer(deLAB_conv_2_cat, out_dim=96, ratio=1, layer_name="toconcat_end_SCA_2_LAB")
    
      # 结束conv 得到 conv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=256)
      deLABconv2_end2_RGB_toRefine = tf.nn.relu(conv2d(toconcat_end_SCA_2_LAB, 96, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv2_end2_RGB_toRefine"))
      deLABconv2_end2_RGB_refine1 = tf.nn.relu(conv2d(deLABconv2_end2_RGB_toRefine, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv2_end2_RGB_refine1"))
      deLABconv2_end2_RGB_refine2 = tf.nn.relu(conv2d(deLABconv2_end2_RGB_refine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv2_end2_RGB_refine2"))
      deLABconv2_end2_RGB_res0 = conv2d(deLABconv2_end2_RGB_refine2, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv2_end2_RGB_res0")
      deLABconv2_end2_RGB_res1 = tf.add(deLABconv2_end2_RGB_res0, deLABconv2_end2_RGB_toRefine)
      deLABconv2_end2_RGB_toRefine1 = tf.nn.relu(conv2d(deLABconv2_end2_RGB_res1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv2_end2_RGB_toRefine1"))
      deLABconv2_end2_RGB_Refine3 = tf.nn.relu(conv2d(deLABconv2_end2_RGB_toRefine1, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv2_end2_RGB_Refine3"))
      deLABconv2_end2_RGB_Refine4 = tf.nn.relu(conv2d(deLABconv2_end2_RGB_Refine3, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv2_end2_RGB_Refine4"))
      deLABconv2_end2_RGB_res2 = conv2d(deLABconv2_end2_RGB_Refine4, 32, 32, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv2_end2_RGB_res2") 
      deLABconv2_end2_RGB_res = tf.add(deLABconv2_end2_RGB_res2,deLABconv2_end2_RGB_toRefine1) 
         
         
# Third RGB decoder
      # 亚像素卷积上采样
      deLAB_conv_3_up = tf.nn.relu(conv2d(deLABconv2_end2_RGB_res, 32, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_3_up"))
      deLAB_conv_3_up2 = conv2d(deLAB_conv_3_up, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1,name="deLAB_conv_3_up2")
      deLAB_conv_3_up3 = tf.nn.depth_to_space(deLAB_conv_3_up2, block_size=2) # 通道数128/4=32
      # 残差连接第0层，融合特征 channel 32 concat 16  -> 48 -> 16
      # deLAB_conv_3_tocat = tf.concat(axis = 3, values = [conv1_RGB, conv1_HSV, conv1_LAB]) # axis=3表示第四个维度进行拼接
      deLAB_conv_3_cat = tf.concat(axis = 3, values = [deLAB_conv_3_up3, conv1_LAB]) # axis=3表示第四个维度进行拼接      
      toconcat_end_SCA_3_LAB = self.Channel_Space_Attention_layer(deLAB_conv_3_cat, out_dim=48, ratio=1, layer_name="toconcat_end_SCA_3_LAB")
      
      # 结束conv 得到 conv1_end1_RGB：(batch_size, height=H/2, width=W/2, channels=256)
      deLABconv3_end3_RGB_toadd = tf.nn.relu(conv2d(toconcat_end_SCA_3_LAB, 48, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv3_end3_RGB_toadd"))
      # deLABconv3_end3_RGB_toRefine = tf.add(deLABconv3_end3_RGB_toadd, conv1_RGB_edge0) # 加入边缘信息
      deLABconv3_end3_RGB_refine1 = tf.nn.relu(conv2d(deLABconv3_end3_RGB_toadd, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv3_end3_RGB_refine1"))
      deLABconv3_end3_RGB_refine2 = tf.nn.relu(conv2d(deLABconv3_end3_RGB_refine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv3_end3_RGB_refine2"))
      deLABconv3_end3_RGB_res0 = conv2d(deLABconv3_end3_RGB_refine2, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv3_end3_RGB_res0")
      deLABconv3_end3_RGB_res1 = tf.add(deLABconv3_end3_RGB_res0, deLABconv3_end3_RGB_toadd)
      deLABconv3_end3_RGB_toRefine1 = tf.nn.relu(conv2d(deLABconv3_end3_RGB_res1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv3_end3_RGB_toRefine1"))
      deLABconv3_end3_RGB_Refine3 = tf.nn.relu(conv2d(deLABconv3_end3_RGB_toRefine1, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv3_end3_RGB_Refine3"))
      deLABconv3_end3_RGB_Refine4 = tf.nn.relu(conv2d(deLABconv3_end3_RGB_Refine3, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv3_end3_RGB_Refine4"))
      deLABconv3_end3_RGB_res2 = conv2d(deLABconv3_end3_RGB_Refine4, 16, 16, k_h=3, k_w=3, d_h=1, d_w=1,name="deLABconv3_end3_RGB_res2") 
      deLABconv3_end3_RGB_res = tf.add(deLABconv3_end3_RGB_res2,deLABconv3_end3_RGB_toRefine1) 
      
      final_results_LAB = tf.nn.sigmoid(conv2d(deLABconv3_end3_RGB_res, 16, 3, k_h=3, k_w=3, d_h=1, d_w=1,name="final_results_LAB"))    
 
 
      # 根据 Retinex 融合
      final_results0 = tf.multiply(tf.nn.relu(final_results_RGB-final_results_noise), final_results_LAB)
      final_results = tf.clip_by_value(final_results0, clip_value_min=0.0, clip_value_max=1.0)
      
      return final_results
      
      
    
  # 保存模型参数：文件夹为 coarse_with_label_height_ + self.label_height
  def save(self, checkpoint_dir, step, ep): # step为已经完成的批次数目，总共有批次数量：训练数据列表的长度 / batch_size(配置中定义的批量大小：config.batch_size)
    model_name = "Multi_ColorSpace_Scale_epoch_" + str(ep) + "---" # 记录已完成epoch数量
    model_dir = "%s_%s" % ("coarse_with_label_height_", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 保存
    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  # 加载
  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse_with_label_height_", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        print("Load success!")
        return True
    else:
        return False


  # SCA 通道空间注意力，输出与input_x相同size的特征
  def Channel_Space_Attention_layer(self, input_x, out_dim, ratio, layer_name):
    # with tf.name_scope(layer_name):
    with tf.compat.v1.variable_scope(layer_name):
      # SCA
      
      # 通道注意力 输出 channel_attention：(batch_size, height=H/2, width=W/2, channels=out_dim（第一层 =128*3=384）)
      # 通道(channel)维度上的全局平均池化
      # keepdims=True：保持维度信息不变，即输出保留通道维度
      channel_mean = tf.reduce_mean(input_x, axis=[1, 2], keepdims=True) 
      # 通道维度上的全局最大池化
      channel_max = tf.reduce_max(input_x, axis=[1, 2], keepdims=True) # (batch_size, height=1, width=1, channels=out_dim（第一层 =128*3=384）)
      # 将全局平均池化和全局最大池化结果拼接在一起：有利于获取整体信息与最显著特征 cb: combine
      channel_cb = tf.concat([channel_mean, channel_max], axis=-1) # -1表示在最后一个维度进行拼接，即channel维度
      # 使用全连接层进行特征转换，并激活函数生成通道注意力权重 cb: combine
      channel_cb_down = tf.keras.layers.Dense(units=int(out_dim / ratio), activation=tf1.nn.gelu, use_bias=False, name="channel_cb_down")(channel_cb) # 输入张量，输出维度，激活函数
      # 全连接层的维度先降低再升高，在一定程度上减少过拟合的风险，同时减少参数数量和模型复杂度，降低总体的计算负担
      channel_attention_weights = tf.keras.layers.Dense(units=out_dim, activation=tf.nn.sigmoid, use_bias=False, name = "channel_attention_weights")(channel_cb_down) # (batch_size, height=1, width=1, channels=out_dim（第一层 =128*3=384）)
      # 将通道注意力权重与输入特征图逐元素相乘 得到(batch_size, 1, 1, channels=input_x的channels（如第一层的SCA输入输出均为384channels）
      channel_attention_weights = tf.reshape(channel_attention_weights, [-1,1,1,out_dim]) # 确保形状一致，可省略 (batch_size, 1, 1, channels=out_dims)
      channel_attention = tf.multiply(input_x, channel_attention_weights) #(batch_size, height=H/2, width=W/2, channels=out_dim（第一层 =128*3=384）
      # 获取张量的形状
      # input_shape = tf.shape(channel_attention)
      # with tf.Session() as sess:
      #   input_shape_value = sess.run(input_shape)
      #   print("channel_attention:", input_shape_value)
          
      
      # 空间注意力 输出SCA最终结果 channel_space_attention: (batch_size, height=H/2, width=W/2, channels=out_dim（第一层 =128*3=384）)
      space_mean = tf.reduce_mean(channel_attention, axis=3, keepdims=True) # keepdims=True：维度信息不变
      space_max = tf.reduce_max(channel_attention, axis=3, keepdims=True) # (batch_size, height=H/2, width=W/2, channels=1)
      # 将全局平均池化和全局最大池化结果拼接在一起：有利于获取整体信息与最显著特征
      space_cb = tf.concat([space_mean, space_max], axis=-1) # -1表示在最后一个维度进行拼接，即channel维度
      # # 第一版：使用全连接层进行特征转换，并激活函数生成通道注意力权重 # (batch_size, height=H/2, width=W/2, channels=1)
      # space_attention_weights = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(space_cb) # 输入张量，输出通道数，激活函数
      # # 第一版
      # 第二版
      space_cb_conv = tf1.nn.gelu(conv2d(space_cb, 2, 1, k_h=3, k_w=3, d_h=1, d_w=1,name="space_combine_conv"))
      space_attention_weights = tf.nn.sigmoid(conv2d(space_cb_conv, 1, 1, k_h=3, k_w=3, d_h=1, d_w=1,name="space_attention_weights"))
      # 第二版

      # # 可加可省略
      # input_shape = tf.shape(input_x) # 获取输入张量的形状
      # h = 1
      # w = 1
      # with tf.Session() as sess:
      #   input_shape_value = sess.run(input_shape)
      #   h = input_shape_value[1]
      #   w = input_shape_value[2]
      #   print("input_x:",input_shape_value)
      # space_attention_weights = tf.reshape(space_attention_weights, [-1,h,w,1]) # 确保形状一致
      # # 可省略   
      
      # 将通道注意力权重与输入特征图逐元素相乘 
      channel_space_attention = tf.multiply(channel_attention, space_attention_weights)
      # SCA 输出 channel_space_attention：(batch_size, height=H/2, width=W/2, channels=out_dim（第一层 =128*3=384）)
      
      return channel_space_attention
    

  # def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
  #   with tf.name_scope(layer_name) :
  #       squeeze = Global_Average_Pooling(input_x)

  #       excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
  #       excitation = Relu(excitation)
  #       excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
  #       excitation = Sigmoid(excitation)

  #       excitation = tf.reshape(excitation, [-1,1,1,out_dim])

  #       scale = input_x * excitation

  #       return scale