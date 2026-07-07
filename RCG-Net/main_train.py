
from model_train import T_CNN
from utils import (
  imsave,
  prepare_data
)
import numpy as np
import tensorflow.compat.v1 as tf

import pprint
import os



flags = tf.app.flags
flags.DEFINE_integer("epoch", 150, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 5, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 256, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 256, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 256 ,"The size of label to produce [230]")
flags.DEFINE_integer("label_width", 256, "The size of label to produce [310]")

flags.DEFINE_float("learning_rate", 0.0001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("checkpoint_dir", "New4_4_C1_4__Batch5_0.0001_0.9_256_bias0.02", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("checkpoint_dir_best", "checkpoint_best__New4_4_C1_4", "Name of checkpoint directory [checkpoint_best]")
flags.DEFINE_string("middle_save", "middle_save_enhance_bias0.02__New4_4_C1_4", "Name of sample directory [middle_save]")
flags.DEFINE_string("final_save", "val_save_enhance__New4_4_C1_4", "Name of sample directory [final_save]")
flags.DEFINE_string("test_data_dir", "test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS


pp = pprint.PrettyPrinter()

def main(_):
  #  # 设置环境变量
  # os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
  # 检查是否存在可用的 GPU
  gpu_available = tf.test.is_gpu_available()

  # 设置 TensorFlow 的配置
  config = tf.ConfigProto()
  if gpu_available:
    config.gpu_options.allow_growth = True  # 允许 GPU 内存按需增长
    config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 设置 GPU 内存使用比例
    device = "/gpu:0"  # 指定使用 GPU
    print("run with GPU")
  else:
    device = "/cpu:0"  # 指定使用 CPU
    print("No GPU, run with CPU")

  # main
  pp.pprint(flags.FLAGS.__flags)
 
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.middle_save):
    os.makedirs(FLAGS.middle_save)
  if not os.path.exists(FLAGS.final_save):
    os.makedirs(FLAGS.final_save)
  if not os.path.exists(FLAGS.checkpoint_dir_best):
    os.makedirs(FLAGS.checkpoint_dir_best)
  
  with tf.Session() as sess:
    with tf.device(device):
      srcnn = T_CNN(sess, 
                  image_height=FLAGS.image_height,
                  image_width=FLAGS.image_width, 
                  label_height=FLAGS.label_height, 
                  label_width=FLAGS.label_width, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  checkpoint_dir_best=FLAGS.checkpoint_dir_best,
                  middle_save=FLAGS.middle_save,
                  final_save=FLAGS.final_save
                  )

      srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
