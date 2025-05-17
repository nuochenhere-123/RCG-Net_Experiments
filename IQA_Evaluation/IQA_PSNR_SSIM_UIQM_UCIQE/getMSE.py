import os
import cv2
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def rmetrics(a,b):
    
    # 防止图片大小不一样报错
    resized_image = cv2.resize(a, (b.shape[1], b.shape[0]))
    a = resized_image
    
    #pnsr
    psnr = compare_psnr(a,b)
    # mse = np.mean(np.square(a-b))
    # psnr = 10. * np.log10(np.square(255.) / mse)

    #ssim
    ssim = compare_ssim(a,b,win_size=3,channel_axis=True)
    # SSIM函数中的win_size参数设置为一个较小的奇数值，以便在计算结构相似性时覆盖图像的局部区域。
    # 一般建议选择一个比较小的值，以确保在图像的不同区域都能得到适当的比较，同时避免在计算时引入太多的噪声或失真。
    # 常见的选择包括3x3、5x5或7x7的窗口大小。
    # 可以根据你的图像大小和特性来进行调整和尝试，以找到最适合的窗口大小。
    # 在实际应用中，通常会根据图像的特点进行调整和优化。
    
    return psnr, ssim

def compute_mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

# 文件夹路径
# 文件夹路径
folder_U45 = './Ucolor_noise-110-256/110-256-poisson'
folder_GT = './110_reference_256'

# 匹配所有图像
img_paths = sorted(glob(os.path.join(folder_U45, '*')))
mse_list = []
psnr_list = []
ssim_list = []
sum = 0

for path_U45 in img_paths:
    filename = os.path.basename(path_U45)
    filename = filename.replace('_out', '') # 处理我们的RCG-Net的输出
    filename = os.path.splitext(filename)[0] + '.png'  # 变成 "image123.png"
    path_GT = os.path.join(folder_GT, filename)
    sum += 1
    if not os.path.exists(path_GT):
        print(f"Warning: {filename} not found in {folder_GT}")
        continue
    print(sum, ": ", filename)

    # 读取图像（BGR模式）
    img_U45 = cv2.imread(path_U45)
    img_GT = cv2.imread(path_GT)

    # 检查尺寸是否一致
    if img_U45.shape != img_GT.shape:
        print(f"Warning: size mismatch for {filename}")
        continue

    # mse = compute_mse(img_U45, img_GT)
    psnr, ssim = rmetrics(img_U45, img_GT)
    # mse_list.append(mse)
    psnr_list.append(psnr)
    ssim_list.append(ssim)

# 计算MSE均值
if psnr_list:
    # mean_mse = np.mean(mse_list)
    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    print(folder_U45, ": ")
    # print(f"Average MSE over {len(mse_list)} image pairs: {mean_mse:.4f}")
    print(f"Average PSNR over {len(psnr_list)} image pairs: {mean_psnr:.6f}, SSIM: {mean_ssim:.6f} ")
else:
    print("No valid image pairs found.")
