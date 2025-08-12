import os
import cv2

def split_rgb_channels(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue
        if '_Rsum' not in os.path.splitext(filename)[0]:
            continue

        # 读取图像
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None or img.ndim != 3 or img.shape[2] != 3:
            print(f"跳过非RGB图像：{filename}")
            continue

        # 提取 RGB 通道（OpenCV读取的是BGR格式）
        B, G, R = cv2.split(img)

        base_name = os.path.splitext(filename)[0]  # 去除扩展名
        # 构造保存路径并写入图像
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_R.png"), R)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_G.png"), G)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_B.png"), B)

        print(f"处理完成：{filename}")

# 示例用法
input_folder = "./res-110"   # 替换为你的输入路径
output_folder = "./res-110"      # 替换为你的输出路径
split_rgb_channels(input_folder, output_folder)
