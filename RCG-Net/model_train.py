from utils import ( 
  imsave,
  prepare_data
)

import matplotlib.pyplot as plt
import wandb
import time
import logging
import shutil
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf1
import tensorflow.compat.v1 as tf
import scipy.io as scio
from ops import *
from ssim import *
from utils import *
from FFL_Loss import FocalFrequencyLoss
# local library
import rgb_lab_formulation as Conv_img
import vgg
class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=128,
               image_width=128,
               label_height=128, 
               label_width=128,
               batch_size=5,
               c_dim=3, 
               checkpoint_dir=None, 
               checkpoint_dir_best=None, 
               sample_dir=None,
               middle_save=None,
               final_save=None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.dropout_keep_prob=0.5
    self.batch_size = batch_size
    self.c_dim = c_dim
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_dir_best = checkpoint_dir_best
    self.middle_save = middle_save
    self.final_save = final_save
    self.sample_dir = sample_dir
    # 
    self.df_dim = 64
    self.vgg_dir='vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    self.CONTENT_LAYER = 'relu5_4'
    self.build_model()


  def build_model(self):
    self.images       = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.labels_image = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')
    self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.images_histeq = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_histeq')
    # self._depth = tf.placeholder(tf.float32, [self.batch_size, self.label_height, self.label_width, 1], name='depth_train')

  
    self.images_test       = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test')
    self.labels_test_image = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, self.c_dim], name='labels_test_image')
    self.images_test_wb       = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_wb')
    self.images_test_gc       = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_gc')
    self.images_test_histeq       = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_histeq')
    # self._test_depth = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, 1], name='test_depth')


    self.raw = self.model()
    # 获取当前图中所有可训练的变量：这些变量通常是神经网络中的权重和偏置等参数，可以在训练过程中通过优化算法来更新它们的值
    t_vars = tf.trainable_variables()
    # 所有含 fution 的字符串变量
    self.fusion_var = [var for var in t_vars if 'fusion' in var.name]
    self.saver = tf.train.Saver()

    # 计算损失
    
    # L2损失：MSE损失
    # self.MSE = 10 * tf.reduce_mean(tf.square(self.labels_image-self.raw))
    self.lambda_MSE = 1.0  # 你可以根据需要调整这个值
    self.MSE=tf.reduce_mean(tf.square(255*(self.labels_image-self.raw)))    
    
    # # Charbonnier 损失
    # self.lambda_char = 10.0  # 你可以根据需要调整这个值
    # self.eps = 0.001
    # self.Charbonnier=tf.reduce_mean(tf.sqrt(tf.square(self.labels_image-self.raw)+tf.square(self.eps)))

    # 感知损失
    self.lambda_VGG = 0.05 # 你可以根据需要调整这个值
    self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image* 255))
    self.raw_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.raw * 255))
    # self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image))
    # self.raw_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.raw))
    self.loss_vgg_raw = tf.reduce_mean(tf.square(self.raw_texture_vgg[self.CONTENT_LAYER]-self.labels_texture_vgg[self.CONTENT_LAYER]))

    # FFL_Loss
    # 转换形状从 (N, H, W, C) 到 (N, C, H, W)
    images_tensor = tf.transpose(self.raw * 255, (0, 3, 1, 2))
    images_tensor2 = tf.transpose(self.labels_image * 255, (0, 3, 1, 2))

    self.lambda_ffl = 0.0015 # 你可以根据需要调整这个值
    ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)  # 初始化FocalFrequencyLoss类
    self.loss_ffl = ffl(images_tensor, images_tensor2)  # 计算损失

    # # L1损失
    # self.l1 = 0.0
    # # 计算绝对差值
    # abs_diff = tf.abs(self.labels_image - self.raw)
    # # 使用 tf.where 计算 L1 损失
    # smoothl1 = tf.where(
    #         tf.less(abs_diff, 1),
    #         0.5 * tf.square(abs_diff),
    #         abs_diff - 0.5
    #     )
    # self.l1 = tf.reduce_mean(smoothl1)

   
    # # UICM损失
    # self.lambda_UICM = 1.0  # You can adjust this weight
    # #1st term UICM
    # rg = self.raw[:,:,:,0] - self.raw[:,:,:,1]
    # yb = (self.raw[:,:,:,0] + self.raw[:,:,:,1]) / 2 - self.raw[:,:,:,2]
    # # 使用 tf.boolean_mask() 获取非 None 值
    # rg_filtered = tf.boolean_mask(rg, tf.math.not_equal(rg, -1))
    # yb_filtered = tf.boolean_mask(yb, tf.math.not_equal(yb, -1))
    # # 对过滤后的张量进行排序
    # rgl = tf.sort(rg_filtered)
    # ybl = tf.sort(yb_filtered)
    # # rgl = tf.sort(rg,axis=None)
    # # ybl = tf.sort(yb,axis=None)
    # al1 = 0.1
    # al2 = 0.1
    # T1 = tf.cast(tf.round(al1 * tf.cast(tf.size(rgl), tf.float32)), tf.int32)
    # T2 = tf.cast(tf.round(al2 * tf.cast(tf.size(rgl), tf.float32)), tf.int32)
    # rgl_tr = rgl[T1:-T2]
    # ybl_tr = ybl[T1:-T2]

    # urg = tf.reduce_mean(rgl_tr)
    # s2rg = tf.reduce_mean(tf.square(rgl_tr - urg))
    # uyb = tf.reduce_mean(ybl_tr)
    # s2yb = tf.reduce_mean(tf.square(ybl_tr- uyb))
    # self.loss_uicm = -0.0268 * tf.sqrt(urg**2 + uyb**2) + 0.1586 * tf.sqrt(s2rg + s2yb)


    # # L2正则化
    # # 设置L2正则化系数
    # self.lambda_l2 = 0.01  # 你可以根据需要调整这个值
    # # 计算所有可训练变量的L2正则化项
    # l2_regularizer = tf.add_n([tf.nn.l2_loss(var) for var in t_vars if 'bias' not in var.name])  # 通常不对偏置项使用L2正则化

    # 添加L2正则化项到损失函数: MSE+0.01L2
    # self.loss = self.lambda_MSE * self.MSE + self.lambda_VGG * self.loss_vgg_raw + self.lambda_l2 * l2_regularizer - self.lambda_UICM * self.loss_uicm
    # self.loss = self.lambda_MSE * self.MSE + self.lambda_VGG * self.loss_vgg_raw + self.lambda_l2 * l2_regularizer - self.lambda_UICM * self.loss_uicm
    # self.loss = self.lambda_MSE * self.MSE + self.lambda_l2 * l2_regularizer
    self.loss = self.lambda_MSE * self.MSE + self.lambda_VGG * self.loss_vgg_raw + self.lambda_ffl * self.loss_ffl 
    #t_vars = tf.trainable_variables()

    self.saver = tf.train.Saver(max_to_keep=0)
    # self.saver = tf.train.Saver(var_list=self.fusion_var)
   
    
  def train(self, config):
    # 准备训练数据
    if config.is_train:     
      # 训练集
      data_train_list, data_train_list_wb, data_train_list_gc, data_train_list_histeq   = prepare_data(self.sess, dataset="raw-780", isRawImages=True) # UIEB-860 + UFO-40
      image_train_list = prepare_data(self.sess, dataset="reference-780", isRawImages=False)
      # 验证集
      data_test_list, data_test_list_wb, data_test_list_gc, data_test_list_histeq = prepare_data(self.sess, dataset="raw-60", isRawImages=True) # UIEB-60
      image_test_list = prepare_data(self.sess, dataset="reference-60", isRawImages=False)

      # 检查
      # print("length!!!!!!!!!!!", suma, sumb, sumc, sumd)
      # # 检查读取后的文件名
      # print("data_train_list:")
      # for file in data_train_list[:10]:  # 只打印前10个文件名以节省空间
      #     print(os.path.basename(file))

      # print("\nimage_train_list:")
      # for file in image_train_list[:10]:
      #     print(os.path.basename(file))

      # 随机打乱，但确保标签数据相互对应
      seed = 1024
      np.random.seed(seed) # 不设置seed时默认使用系统时间作为种子值，每次运行时种子都会不同，因此生成的随机数序列也会不同
      np.random.shuffle(data_train_list)
      np.random.seed(seed) # 重置随机种子，确保两次打乱采用相同的种子值
      np.random.shuffle(data_train_list_wb)
      np.random.seed(seed) # 重置随机种子，确保两次打乱采用相同的种子值
      np.random.shuffle(data_train_list_gc)
      np.random.seed(seed) # 重置随机种子，确保两次打乱采用相同的种子值
      np.random.shuffle(data_train_list_histeq)
      np.random.seed(seed) # 重置随机种子，确保两次打乱采用相同的种子值
      np.random.shuffle(image_train_list)
      # 确保数据集的顺序打乱，但是标签与数据仍然一一对应

      # # 检查排序后的文件名
      # print("data_train_list:")
      # for file in data_train_list[:10]:  # 只打印前10个文件名以节省空间
      #     print(os.path.basename(file))
      # print("\nimage_train_list:")
      # for file in image_train_list[:10]:
      #     print(os.path.basename(file))

    # 准备测试数据
    else:
      data_test_list, data_test_list_wb, data_test_list_gc, data_test_list_histeq = prepare_data(self.sess, dataset="raw-60", isRawImages=True) # UIEB-60
      image_test_list = prepare_data(self.sess, dataset="reference-60", isRawImages=False)

    # 设置wandb
    wandb.init(project = "MSE+0.002FFL+0.05VGG_RES_with_LR*cos_epoch")
    wandb.config.update(config, allow_val_change = True)
    config = wandb.config
   
    # # 最小化损失函数 且 仅优化 含fusion的 可优化变量 (fusion_var在L77)
    # # self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
    # #          .minimize(self.loss,var_list=self.fusion_var)
    # # 原策略
    # self.g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
    #          .minimize(self.loss)
    # # 原策略


    # # 指数衰减策略
    # # 初始学习率
    # initial_learning_rate = config.learning_rate
    # global_step = tf.Variable(0, trainable=False)
    # decay_steps = 780 # 每10个epoch衰减为原来的0.9
    # decay_rate = 0.9
    # # 使用指数衰减函数
    # learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
    #                                           decay_steps, decay_rate, staircase=True)
    # self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=config.beta1) \
    #          .minimize(self.loss, global_step=global_step)
    # # 指数衰减策略
    
    
    # 余弦退火策略
    # 初始学习率
    initial_learning_rate = config.learning_rate
    global_step = tf.Variable(0, trainable=False)
    total_epochs = config.epoch  # 总的训练周期，根据具体情况调整
    total_steps = 156 * total_epochs # 总的训练步数
    # 定义余弦退火调度器：根据余弦函数的特性，学习率会渐渐接近但不会严格等于零，因此没有显式的最小学习率设定
    learning_rate_cosine = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        total_steps, # 学习率从 初始 逐渐衰减到最低点的步数（即总的迭代步骤数），在这个步数内，学习率将经历一个完整的余弦周期
        alpha=0.1,  # 默认为0，表示衰减到0; 0.1表示衰减到初始学习率的10%
    )
    learning_rate = learning_rate_cosine(global_step)
    # 定义优化器
    self.g_optim = tf.train.AdamOptimizer(learning_rate_cosine(global_step), beta1=config.beta1) \
             .minimize(self.loss, global_step=global_step)
    # 余弦退火策略
    
    
    # 初始化所有的全局变量，确保它们可以在后续的计算中被正确使用
    tf.global_variables_initializer().run()
    
    #################################################################################################
    
    counter = 0 # 已完成的批次数量
    no_better_record = 0 # 无优化epoch
    start_time = time.time()
    min_loss_value = 10000.0 # 记录最小loss

    # 1：train 0：test
    if config.is_train:
      print("Training...")
      # 记录每个 epoch 的损失值
      loss_sum = np.ones(config.epoch) # 设置一个长度为 config.epoch 的全是 1 的数组
      loss_char = np.ones(config.epoch)
      loss_vgg = np.ones(config.epoch)
      loss_ffl = np.ones(config.epoch)
      # 每个 epoch 中包含的批次数量：训练数据列表的长度 / batch_size(配置中定义的批量大小)
      batch_idxs = len(data_train_list) // config.batch_size
      
      # 遍历 epoch （ 在 main_train.py 中设置 ）
      for ep in range(config.epoch):
        # if ep == 1 or ep == 0:
        #   ep = 33 # 已经运行了多少epoch就写多少epoch
        # 读取参数
        if self.load(self.checkpoint_dir): 
          print(" [*] Load SUCCESS ", str(self.checkpoint_dir))
        else:
          print(" [!] Load failed...")
        record_char = 0.0
        record_vgg = 0.0
        record_ffl = 0.0
        record_sum = 0.0
        
        # 遍历每个批次
        for idx in range(0, batch_idxs):
          # 选择每个批次中的图片 并 读取 之后 转换类型
          print("start ",idx," in ep ",ep," with total ",batch_idxs)
          # 选择
          batch_files         = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size] # raw
          batch_files_wb      = data_train_list_wb[idx*config.batch_size:(idx+1)*config.batch_size] # gc
          batch_files_gc      = data_train_list_gc[idx*config.batch_size:(idx+1)*config.batch_size] # gc
          batch_files_histeq  = data_train_list_histeq[idx*config.batch_size:(idx+1)*config.batch_size] # histeq
          # print("batch_files:", batch_files)
          batch_image_files        = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size] # reference
          # print("batch_image_files:", batch_image_files)
          # batch_depth_files = depth_train_list[idx*config.batch_size : (idx+1)*config.batch_size]
                    
          batch_input = []
          batch_input_wb = []
          x_offsets = []
          y_offsets = []
          lr_now = 0.0

          # 连续读取 raw 在 get_image 中 读取一张图片，并随机裁剪，转换类型float32
          for batch_file in batch_files:
              img, x_offset, y_offset = get_image_raw(batch_file, is_grayscale=self.is_grayscale)
              batch_input.append(img)
              x_offsets.append(x_offset)
              y_offsets.append(y_offset)

          # 连续获取白平衡图像，确保亮度
          batch_input_wb = []
          for batch_file_wb, x_offset, y_offset in zip(batch_files_wb, x_offsets, y_offsets):
              img_wb = get_image_reference_wb_gc_histeq(batch_file_wb, x_offset=x_offset, y_offset=y_offset, is_grayscale=self.is_grayscale)
              batch_input_wb.append(img_wb)
              
           # 连续获取伽马校正图像，确保亮度
          batch_input_gc = []
          for batch_file_gc, x_offset, y_offset in zip(batch_files_gc, x_offsets, y_offsets):
              img_gc = get_image_reference_wb_gc_histeq(batch_file_gc, x_offset=x_offset, y_offset=y_offset, is_grayscale=self.is_grayscale)
              batch_input_gc.append(img_gc)
              
          # 连续获取直方图均衡化图像，确保对比度
          batch_input_histeq = []
          for batch_file_histeq, x_offset, y_offset in zip(batch_files_histeq, x_offsets, y_offsets):
              img_histeq = get_image_reference_wb_gc_histeq(batch_file_histeq, x_offset=x_offset, y_offset=y_offset, is_grayscale=self.is_grayscale)
              batch_input_histeq.append(img_histeq)

          # 连续读取 reference
          batch_image_input = []
          for batch_image_file, x_offset, y_offset in zip(batch_image_files, x_offsets, y_offsets):
              img_ref = get_image_reference_wb_gc_histeq(batch_image_file, x_offset=x_offset, y_offset=y_offset, is_grayscale=self.is_grayscale)
              batch_image_input.append(img_ref)


          # 检查并确保所有图像的形状一致
          for img in batch_input:
              if img.shape != (256, 256, 3):  # 假设图像有1个或3个通道
                  raise ValueError(f"Inconsistent shape {img} in batch_input: {img.shape}")
                
          # 检查并确保所有图像的形状一致
          for img in batch_input_gc:
              if img.shape != (256, 256, 3):  # 假设图像有1个或3个通道
                  raise ValueError(f"Inconsistent shape {img} in batch_input_gc: {img.shape}")
          
          # 检查并确保所有图像的形状一致
          for img in batch_input_wb:
              if img.shape != (256, 256, 3):  # 假设图像有1个或3个通道
                  raise ValueError(f"Inconsistent shape {img} in batch_input_wb: {img.shape}")
                
          # 检查并确保所有图像的形状一致
          for img in batch_input_histeq:
              if img.shape != (256, 256, 3):  # 假设图像有1个或3个通道
                  raise ValueError(f"Inconsistent shape {img} in batch_input_histeq: {img.shape}")
          
          for img in batch_image_input:
              if img.shape != (256, 256, 3):  # 假设图像有1个或3个通道
                  raise ValueError(f"Inconsistent shape {img} in batch_image_input: {img.shape}")

          batch_input = np.array(batch_input)
          batch_input_wb = np.array(batch_input_wb)
          batch_input_gc = np.array(batch_input_gc)
          batch_input_histeq = np.array(batch_input_histeq)
          batch_image_input = np.array(batch_image_input)
 #################################################################################################
    
          # 已完成的该批次的loss（第一个批次结束后，counter为1）
          counter += 1
          
          # # 优化更新模型参数 self.g_optim，监控记录损失值 MSE、vgg_loss、总体loss、每张图像的 encoder_fusion_res 结果
          # _, err1,err2,err3,err4,enhanced_image= self.sess.run([self.g_optim, self.MSE, self.loss_vgg_raw, self.loss_uicm, self.loss, self.raw], feed_dict={self.images: batch_input, self.labels_image:batch_image_input}) 
          # record_char += err1
          # record_vgg += err2
          # record_uicm += err3
          # record_sum += err4
          _, err1,err2,err3,err4,enhanced_image,lr_now= self.sess.run([self.g_optim, self.MSE, self.loss_vgg_raw, self.loss_ffl, self.loss, self.raw, learning_rate], feed_dict={self.images: batch_input, self.images_wb: batch_input_wb, self.images_gc: batch_input_gc, self.images_histeq: batch_input_histeq, self.labels_image:batch_image_input}) 
          record_char += err1
          record_vgg += err2
          record_ffl += err3
          record_sum += err4

          #  # 每个批次记录一次训练集结果: epoch，已经运行的批次数量，时间，损失记录
          # print("Epoch: [%2d], step: [%2d], time: [%4.4f],MSE_train_loss: [%.8f],VGG_train_loss: [%.8f],UICM_train_loss: [%.8f],final_train_loss: [%.8f]" \
          #     % ((ep+1), counter, time.time()-start_time,err1,err2,err3,err4))          
          print("Epoch: [%2d], step: [%2d], time: [%4.4f],MSE_train_loss: [%.8f], VGG_train_loss: [%.8f], FFL_train_loss: [%.8f], final_train_loss: [%.8f], LR: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time,err1,err2,err3,err4,lr_now))
            
          # 保存训练集原始图像、中间结果图像、增强结果图像 
          # 开始遍历批次中的每张图像
          for i in range(len(batch_files)):
            # 保存原始结果
            save_path_original = os.path.join(self.middle_save, f'{idx * config.batch_size + i}_original.png')
            plt.imsave(save_path_original, batch_input[i])
            # # 保存中间结果
            # save_path_middle = os.path.join(self.middle_save, f'{idx * config.batch_size + i}_encoder_fusion_res.png')
            # # combined_feature_map_rgb = np.clip(encoder_fusion_res_results[i], 0, 1)
            # # 多通道图像输出
            # combined_feature_map = np.mean(encoder_fusion_res_results[i], axis=-1)
            # combined_feature_map_rgb = np.stack((combined_feature_map,) * 3, axis=-1)
            # combined_feature_map_rgb = np.clip(combined_feature_map_rgb, 0, 1)
            # plt.imsave(save_path_middle, combined_feature_map_rgb)
            # 保存增强结果
            _,h ,w , c = enhanced_image.shape
            final_enhanced_img = np.clip(enhanced_image[i], 0, 1).reshape(h , w , 3)
            # save_path_en = os.path.join(self.middle_save, f'{idx * config.batch_size + i}_enhance.png')
            # plt.imsave(save_path_en, final_enhanced_img)
            # 或 保存增强结果
            save_path_en_io = os.path.join(self.middle_save, f'{idx * config.batch_size + i}_enhance_io.png')
            # save_path_en_plt = os.path.join(self.middle_save, f'{idx * config.batch_size + i}_enhance_plt.png')
            final_enhanced_img_out = np.uint8(final_enhanced_img*255)
            io.imsave(save_path_en_io, final_enhanced_img_out) 
            # final_enhanced_img_out_1 = Image.fromarray(np.uint8(final_enhanced_img*255))
            # final_enhanced_img_out_1.save(save_path_en_plt)
 
          
          
          # 最后一个批次（结束后该epoch结束）训练完成后：
          # 进行验证集测试
          if idx  == batch_idxs-1:
            # 记录该epoch训练集平均损失
            # print("Epoch: [%2d] ,MSE_sumMean_train_loss: [%.8f],VGG_sumMean_train_loss: [%.8f],UICM_sumMean_train_loss: [%.8f],final_sumMean_train_loss: [%.8f] \n" \
            #   % ((ep+1), record_char/batch_idxs, record_vgg/batch_idxs, record_uicm/batch_idxs, record_sum/batch_idxs))
            # wandb.log({f"[Sum] MSE_epoch_train" : record_char/batch_idxs,
            #            "[Sum] VGG_epoch_train" : record_vgg/batch_idxs,
            #            "[Sum] UICM_epoch_train" : record_uicm/batch_idxs,
            #            "[Sum] Final_epoch_train" : record_sum/batch_idxs,},
            #                commit = True)
            print("Epoch: [%2d], MSE_sumMean_train_loss: [%.8f], VGG_sumMean_train_loss: [%.8f], FFL_sumMean_train_loss: [%.8f], final_sumMean_train_loss: [%.8f] \n" \
              % ((ep+1), record_char/batch_idxs, record_vgg/batch_idxs, record_ffl/batch_idxs, record_sum/batch_idxs))
            wandb.log({f"[Sum] MSE_epoch_train" : record_char/batch_idxs,
                       "[Sum] VGG_epoch_train" : record_vgg/batch_idxs,
                       "[Sum] FFL_epoch_train" : record_ffl/batch_idxs,
                       "[Sum] Final_epoch_train" : record_sum/batch_idxs,},
                           commit = True)
            # 验证集数据列表的长度 / batch_size(配置中定义的批量大小)
            batch_test_idxs = len(data_test_list) // config.batch_size
            # 记录验证集每个批次的loss
            err_sum_test =  np.ones(batch_test_idxs)
            err_char_test =  np.ones(batch_test_idxs)            
            err_vgg_test =  np.ones(batch_test_idxs)
            err_ffl_test =  np.ones(batch_test_idxs)
            # 遍历 验证集 每个批次
            for idx_test in range(0,batch_test_idxs): 
              # 选择验证集中每个批次的图片 并 读取 之后 转换类型

              # 选择：验证集 数据 中 一个 batch_size 的（不包括索引 20）元素
              sample_data_files = data_test_list[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_data_files_wb = data_test_list_wb[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_data_files_gc = data_test_list_gc[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_data_files_histeq = data_test_list_histeq[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_image_files = image_test_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              # sample_image_files1 = depth_test_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
 
              sample_inputs_data = []
              # 裁剪后作验证集
              x_offsets1 = []
              y_offsets1 = []
              # 连续读取 raw 在 get_image 中 读取一张图片，并随机裁剪，转换类型float32
              for sample_data_file in sample_data_files:
                  img1, x_offset1, y_offset1 = get_image_raw(sample_data_file, is_grayscale=self.is_grayscale)
                  sample_inputs_data.append(img1)
                  x_offsets1.append(x_offset1)
                  y_offsets1.append(y_offset1)
              sample_inputs_data = np.array(sample_inputs_data)

              # 连续读取 白平衡 图像
              sample_inputs_data_wb = []
              for sample_data_file_wb, x_offset1, y_offset1 in zip(sample_data_files_wb, x_offsets1, y_offsets1):
                  img_wb1 = get_image_reference_wb_gc_histeq(sample_data_file_wb, x_offset=x_offset1, y_offset=y_offset1, is_grayscale=self.is_grayscale)
                  sample_inputs_data_wb.append(img_wb1)
              sample_inputs_data_wb = np.array(sample_inputs_data_wb)
              
              # 连续读取 伽马校正 图像
              sample_inputs_data_gc = []
              for sample_data_file_gc, x_offset1, y_offset1 in zip(sample_data_files_gc, x_offsets1, y_offsets1):
                  img_gc1 = get_image_reference_wb_gc_histeq(sample_data_file_gc, x_offset=x_offset1, y_offset=y_offset1, is_grayscale=self.is_grayscale)
                  sample_inputs_data_gc.append(img_gc1)
              sample_inputs_data_gc = np.array(sample_inputs_data_gc)
              
              # 连续读取 直方图均衡化 图像
              sample_inputs_data_histeq = []
              for sample_data_file_histeq, x_offset1, y_offset1 in zip(sample_data_files_histeq, x_offsets1, y_offsets1):
                  img_histeq1 = get_image_reference_wb_gc_histeq(sample_data_file_histeq, x_offset=x_offset1, y_offset=y_offset1, is_grayscale=self.is_grayscale)
                  sample_inputs_data_histeq.append(img_histeq1)
              sample_inputs_data_histeq = np.array(sample_inputs_data_histeq)

              # 连续读取 reference
              sample_inputs_lable_image = []
              for sample_image_file, x_offset1, y_offset1 in zip(sample_image_files, x_offsets1, y_offsets1):
                  img_ref1 = get_image_reference_wb_gc_histeq(sample_image_file, x_offset=x_offset1, y_offset=y_offset1, is_grayscale=self.is_grayscale)
                  sample_inputs_lable_image.append(img_ref1)
              sample_inputs_lable_image = np.array(sample_inputs_lable_image)
              

              # 仅监控保存该批次验证集中该批次的loss,验证集不更新模型参数
              err_char_test[idx_test], err_vgg_test[idx_test], err_ffl_test[idx_test], err_sum_test[idx_test],val_encoder_fusion_res_results = self.sess.run([self.MSE, self.loss_vgg_raw, self.loss_ffl, self.loss, self.raw], feed_dict={self.images: sample_inputs_data, self.images_wb: sample_inputs_data_wb, self.images_gc: sample_inputs_data_gc, self.images_histeq: sample_inputs_data_histeq, self.labels_image:sample_inputs_lable_image})    
              
              # 保存验证集增强结果图像 
              for i in range(len(sample_data_files)):
                # 保存原始结果
                save_path_original = os.path.join(self.final_save, f'{idx * config.batch_size + i}_val_original.png')
                plt.imsave(save_path_original, sample_inputs_data[i])
                # 保存增强结果
                _,h1 ,w1 , c1 = val_encoder_fusion_res_results.shape            
                final_enhanced_img = np.clip(val_encoder_fusion_res_results[i], 0, 1).reshape(h1 , w1 , 3)
                save_path_en = os.path.join(self.final_save, f'{idx * config.batch_size + i}_val_enhance.png')
                io.imsave(save_path_en, np.uint8(final_enhanced_img*255))
          
            # 验证集的平均loss，epoch 1 的结果在 loss[0] 中
            loss_char[ep]=np.mean(err_char_test)
            loss_vgg[ep]=np.mean(err_vgg_test)
            loss_ffl[ep]=np.mean(err_ffl_test)
            loss_sum[ep]=np.mean(err_sum_test)
            # 每个epoch记录一次验证集结果: epoch，已经运行的批次数量，时间，损失记录
            # print("Epoch: [%2d], step: [%2d], time: [%4.4f],MSE_val_loss: [%.8f],VGG_val_loss: [%.8f],uicm_val_loss: [%.8f],final_val_loss: [%.8f]" \
            #   % ((ep+1), counter, time.time()-start_time,loss_char[ep],loss_vgg[ep],loss_uicm[ep],loss_sum[ep]))
            # wandb.log({f"[Val] MSE val_Loss" : loss_char[ep],
            #            "[Val] VGG val_Loss" : loss_vgg[ep],
            #            "[Val] UICM val_Loss" : loss_uicm[ep],
            #            "[Val] Final val_Loss" : loss_sum[ep],},
            #              commit = True)
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], MSE_val_loss: [%.8f], VGG_val_loss: [%.8f], FFL_val_loss: [%.8f], final_val_loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time,loss_char[ep],loss_vgg[ep],loss_ffl[ep],loss_sum[ep]))
            wandb.log({f"[Val] MSE val_Loss" : loss_char[ep],
                       "[Val] VGG val_Loss" : loss_vgg[ep],
                       "[Val] FFL val_Loss" : loss_ffl[ep],
                       "[Val] Final val_Loss" : loss_sum[ep],},
                         commit = True)
            # 若loss更小，保存该epoch结束后的模型参数
            if os.path.exists(config.checkpoint_dir):
                shutil.rmtree(config.checkpoint_dir)  # 删除目录及其所有内容
            os.makedirs(config.checkpoint_dir)  # 重新创建目录
            self.save(config.checkpoint_dir, counter, ep+1)
            
            if loss_sum[ep] <= min_loss_value:
              min_loss_value = loss_sum[ep]
              no_better_record = 0
              if os.path.exists(config.checkpoint_dir_best):
                shutil.rmtree(config.checkpoint_dir_best)  # 删除目录及其所有内容
                os.makedirs(config.checkpoint_dir_best)  # 重新创建目录
              self.save(config.checkpoint_dir_best, counter, ep+1)
            else:
              no_better_record += 1
              if no_better_record >= 10:
                print(f"Early stopping triggered after {ep+1} epochs with best validation loss: {min_loss_value}")
                break
              

  # 网络结构
  def model(self):
    with tf.variable_scope("fusion_branch") as scope1: 
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
    with tf.variable_scope(layer_name):
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
    