from model_test import T_CNN
from utils import *
import numpy as np
import tensorflow.compat.v1 as tf

import glob
import cv2
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


def extract_frames(video_path, frame_dir):
    """Extract frames from a video and save them as images."""
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    height = 0 
    width = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = os.path.join(frame_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_name, frame)
        frame_count += 1
    print("Total need to process: ", frame_count, "!!!!!!!!!!")
    cap.release()
    return frame_dir, frame_count

def combine_frames_to_video(frame_dir, output_video_path, original_fps):
    """Combine processed frames into a video."""
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*_out.png")))
    if len(frame_files) == 0:
        print("No frames found to combine!")
        return

    # Read the first frame to get video dimensions
    frame = cv2.imread(frame_files[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, original_fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()


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
  root_dir = os.path.join(os.getcwd(), 'Test')

  # 遍历 Test 文件夹下的所有子文件夹
  for data_dir, _, _ in os.walk(root_dir):
    # data_dir = os.path.join(os.getcwd(), 'Test-R90') # input image dataset
    
    # 获取 png 图片列表 和 jpg 图片列表： 使用 glob.glob() 根据特定的文件扩展名获取文件列表
    # test_data_list = sorted(glob.glob(os.path.join(data_dir, "*.png"))) + sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    video_files = sorted(glob.glob(os.path.join(data_dir, "*.mp4")) + glob.glob(os.path.join(data_dir, "*.avi")))
   
    for video_file in video_files:
      print(f"Processing video: {video_file}")

      # Step 1: Extract frames          
      frame_dir = os.path.join(data_dir, "frames_" + os.path.basename(video_file).split('.')[0])
      frame_dir, frame_count = extract_frames(video_file, frame_dir)

      # Step 2: Process frames
      # 获取 png 图片列表 和 jpg 图片列表： 使用 glob.glob() 根据特定的文件扩展名获取文件列表
      test_data_list = sorted(glob.glob(os.path.join(frame_dir, "*.png"))) + sorted(glob.glob(os.path.join(frame_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(frame_dir, "*.jpeg")))

      for ide in range(0,len(test_data_list)):
        image_test1 =  get_image_original(test_data_list[ide],is_grayscale=False)
        # image_test1 = cv2.resize(image_test1, (256,256))
        shape = image_test1.shape
        RGB=Image.fromarray(np.uint8(image_test1*255))
        RGB1=RGB.resize(((shape[1]//8-0)*8,(shape[0]//8-0)*8))
        image_test = np.asarray(np.float32(RGB1)/255)
      
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

      # Step 3: Combine processed frames into a video
      output_video_path = os.path.join(data_dir, "processed_" + os.path.basename(video_file))
      original_fps = int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FPS))
      combine_frames_to_video(frame_dir, output_video_path, original_fps)

 
if __name__ == '__main__':
  tf.app.run()
