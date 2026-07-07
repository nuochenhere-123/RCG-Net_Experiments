from model_test import T_CNN
from utils import *
import numpy as np
import tensorflow.compat.v1 as tf

import pprint
import os
import PIL

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 256, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 256, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 256 ,"The size of label to produce [230]")
flags.DEFINE_integer("label_width", 256, "The size of label to produce [310]")

flags.DEFINE_float("learning_rate", 0.0001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("counter", 1, "sum of test image already")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_best__New4_4_C1_4", "Name of checkpoint directory [checkpoint]")
# flags.DEFINE_string("checkpoint_dir", "New4_4_C1_8__Batch5_0.0001_0.9_256_bias0.02", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("test_data_dir", "test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.99  # 设置 GPU 内存使用比例
    device = "/gpu:0"  # 指定使用 GPU
    print("run with GPU")
  else:
    device = "/cpu:0"  # 指定使用 CPU
    print("No GPU, run with CPU")

  # main
  pp.pprint(flags.FLAGS.__flags)
 
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  # 根目录
  root_dir = os.path.join(os.getcwd(), 'ImageForTest')

  # 遍历 Test 文件夹下的所有子文件夹
  for data_dir, _, _ in os.walk(root_dir):
    # data_dir = os.path.join(os.getcwd(), 'Test-R90') # input image dataset
    
    # 获取 png 图片列表 和 jpg 图片列表： 使用 glob.glob() 根据特定的文件扩展名获取文件列表
    # test_data_list = sorted(glob.glob(os.path.join(data_dir, "*.png"))) + sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    test_data_list = sorted(glob.glob(os.path.join(data_dir, "*.png"))) + sorted(glob.glob(os.path.join(data_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(data_dir, "*.jpeg")))

    # filenames1 = os.listdir('real_90_gdcp') ###### input transmission dataset
    # data_dir1 = os.path.join(os.getcwd(), 'real_90_gdcp')
    # data1 = sorted(glob.glob(os.path.join(data_dir1, "*.png")))
    # test_data_list1 = data1 + sorted(glob.glob(os.path.join(data_dir1, "*.jpg")))+sorted(glob.glob(os.path.join(data_dir1, "*.bmp")))+sorted(glob.glob(os.path.join(data_dir1, "*.jpeg")))

    for ide in range(0,len(test_data_list)):
      image_test1 =  get_image_original(test_data_list[ide],is_grayscale=False)
      image_test1 = cv2.resize(image_test1, (256,256))
      shape = image_test1.shape
      RGB=Image.fromarray(np.uint8(image_test1*255))
      RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
      image_test = np.asarray(np.float32(RGB1)/255)
      # depth_test1 =  get_image(test_data_list1[ide],is_grayscale=False)
      # Depth=Image.fromarray(np.uint8(depth_test1*255))
      # Depth1=Depth.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
      # depth_test = np.asarray(np.float32(Depth1)/255)

      shape = image_test.shape
      tf.reset_default_graph()
      with tf.Session() as sess:
        # with tf.device('/cpu:0'):
        with tf.device(device):
          srcnn = T_CNN(sess, 
                    image_height=shape[0],
                    image_width=shape[1],  
                    label_height=FLAGS.label_height, 
                    label_width=FLAGS.label_width, 
                    batch_size=FLAGS.batch_size,
                    counter = FLAGS.counter,
                    c_dim=FLAGS.c_dim, 
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    test_image_name = test_data_list[ide],
                    id = ide
                    )

          srcnn.train(FLAGS)
          sess.close()
      tf.get_default_graph().finalize()
      
if __name__ == '__main__':
  tf.app.run()
