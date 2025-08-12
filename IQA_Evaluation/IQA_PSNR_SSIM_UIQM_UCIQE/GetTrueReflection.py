import os
import numpy as np
from PIL import Image

# 设置文件夹路径
folder = "./Output"
output_suffixes = ["R", "G", "B"]
sum_suffixes = [f"Rsum_{c}" for c in output_suffixes]
noise_suffixes = [f"noise_{c}" for c in output_suffixes]
true_suffixes = [f"Rtrue_{c}" for c in output_suffixes]

# 确保文件夹存在
if not os.path.exists(folder):
    print(f"Folder '{folder}' does not exist.")
    exit()

files = os.listdir(folder)
print(f"Total files found in '{folder}':", len(files))

# 遍历每个颜色通道
for color, sum_suf, noise_suf, true_suf in zip(output_suffixes, sum_suffixes, noise_suffixes, true_suffixes):
    # 查找所有以 R_sum-R/G/B 结尾的文件
    for file in files:
        if file.endswith(f"{sum_suf}.png"):
            prefix = file[:-len(f"{sum_suf}.png")]
            sum_path = os.path.join(folder, f"{prefix}{sum_suf}.png")
            noise_path = os.path.join(folder, f"{prefix}{noise_suf}.png")
            true_path = os.path.join(folder, f"{prefix}{true_suf}.png")

            # 检查对应的 noise 是否存在
            if not os.path.exists(noise_path):
                print(f"Missing noise image for: {file}")
                continue

            # 加载图像为 float32
            sum_img = np.array(Image.open(sum_path)).astype(np.float32) / 255.0
            noise_img = np.array(Image.open(noise_path)).astype(np.float32) / 255.0

            # 相减并裁剪到 [0, 1]
            true_img = np.clip(sum_img - noise_img, 0.0, 1.0)
            true_img_uint8 = (true_img * 255).astype(np.uint8)

            # 保存
            Image.fromarray(true_img_uint8).save(true_path)
            print(f"Saved: {true_path}")

print("Done.")
