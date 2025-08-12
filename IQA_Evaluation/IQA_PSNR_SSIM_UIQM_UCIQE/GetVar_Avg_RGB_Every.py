# 获取 某个文件夹内所有图像的 RGB 三通道方差均值 的 直方图分布
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def compute_channel_variance_per_image(image_folder, IsOrNot = False):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    all_var_r, all_var_g, all_var_b = [], [], []
    var_mean_list = []
    mean_var_per_image = []  # 存储每张图像的均值和文件名

    print("每张图像的通道方差信息：\n")

    for path in image_paths:
        image_name = os.path.basename(path)
        image = cv2.imread(path)  # BGR
        if image is None:
            print(f"跳过无法读取的图像: {path}")
            continue
        
        if IsOrNot:
            image = cv2.resize(image, (256, 256))

        var_b = np.var(image[:, :, 0])
        var_g = np.var(image[:, :, 1])
        var_r = np.var(image[:, :, 2])
        var_mean = (var_r + var_g + var_b) / 3

        all_var_b.append((image_name, var_b))
        all_var_g.append((image_name, var_g))
        all_var_r.append((image_name, var_r))
        var_mean_list.append(var_mean)
        mean_var_per_image.append((image_name, var_mean))

        print(f"{image_name}: R_var={var_r:.2f}, G_var={var_g:.2f}, B_var={var_b:.2f}, Mean_var={var_mean:.2f}")

    def format_extremes(data, name, mean_val):
        sorted_data = sorted(data, key=lambda x: x[1])
        s = f"\n{name}通道平均方差: {mean_val:.2f}\n  最大三值："
        for n, v in sorted(data[-3:], key=lambda x: -x[1]):
            s += f"\n    {n}: {v:.2f}"
        s += f"\n  最小三值："
        for n, v in sorted_data[:3]:
            s += f"\n    {n}: {v:.2f}"
        return s

    r_mean = np.mean([x[1] for x in all_var_r])
    g_mean = np.mean([x[1] for x in all_var_g])
    b_mean = np.mean([x[1] for x in all_var_b])
    global_mean = np.mean(var_mean_list)

    print(format_extremes(all_var_r, "R", r_mean))
    print(format_extremes(all_var_g, "G", g_mean))
    print(format_extremes(all_var_b, "B", b_mean))

    # 方差均值最大/最小三图
    sorted_mean_var = sorted(mean_var_per_image, key=lambda x: x[1])
    print(f"\n通道方差均值的全局平均: {global_mean:.2f}\n  最大三值：")
    for n, v in sorted(sorted_mean_var[-3:], key=lambda x: -x[1]):
        print(f"    {n}: {v:.2f}")
    print("  最小三值：")
    for n, v in sorted_mean_var[:3]:
        print(f"    {n}: {v:.2f}")

    # 绘制分布直方图
    plt.figure(figsize=(8, 5))
    bins = np.arange(0, max(var_mean_list) + 200, 200)
    plt.hist(var_mean_list, bins=bins, edgecolor='black', color='skyblue')
        
    # 计算低于5000的比例
    below_5000 = sum(v < 5000 for v in var_mean_list)
    total = len(var_mean_list)
    percentage = (below_5000 / total) * 100
    # 计算 95% 处的值
    percentile_95 = np.percentile(var_mean_list, 95)
    
    # 确保输出目录存在
    folder_new = "var-every"
    os.makedirs(folder_new, exist_ok=True)

    # 设置支持中文的字体（如 SimHei 或 Microsoft YaHei）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    name_now = os.path.basename(os.path.normpath(image_folder))
    if IsOrNot:
        plt.title(f"{name_now}-256 中 单张图像 的 三通道方差均值分布 直方图")
    else:
        plt.title(f"{name_now} 中 单张图像 的 三通道方差均值分布 直方图")
    plt.xlabel(f"方差 95%处值为 {percentile_95:.2f}, 通道方差均值 < 5000 占比：{percentage:.2f}% ")
    plt.ylabel("图像数量")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if IsOrNot:
        plt.savefig(f"./var-every/var-{name_now}-256.png")
    else:
        plt.savefig(f"./{folder_new}/var-{name_now}.png")
    print(f"\n分布直方图已保存为 var.png, 95%处值为 {percentile_95:.2f}, 方差 < 5000 的图像占比为 {percentage:.2f}%")
    
    suffix = "-256" if IsOrNot else ""
    # ✅ R/G/B 单独通道的分布直方图
    def draw_channel_hist(channel_data, channel_name, color, below):
        # 提取方差值列表
        values = [v for _, v in channel_data]
        # 计算低于5000的比例
        below_5000 = sum(v < below for v in values)
        total = len(values)
        percentage = (below_5000 / total) * 100
        # ✅ 95% 处值
        percentile_95 = np.percentile(values, 95)  
        
        values = [v for _, v in channel_data]
        plt.figure(figsize=(8, 5))
        bins = np.arange(0, max(values) + 200, 200)
        plt.hist(values, bins=bins, edgecolor='black', color=color)
        plt.title(f"{name_now}{suffix} 中 单张图像 的 {channel_name} 通道方差分布")
        plt.xlabel(f"方差 95%处值为 {percentile_95:.2f}, 通道方差均值 < {below} 占比：{percentage:.2f}% ")
        plt.ylabel("图像数量")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename = f"./{folder_new}/var-{name_now}{suffix}-{channel_name}.png"
        plt.savefig(filename)
        print(f"{channel_name} 通道直方图已保存为 {filename}, 95%处值为 {percentile_95:.2f}, 方差 < {below} 的图像占比为 {percentage:.2f}%")

    draw_channel_hist(all_var_r, "R", "salmon", 6000)
    draw_channel_hist(all_var_g, "G", "mediumseagreen", 5500)
    draw_channel_hist(all_var_b, "B", "cornflowerblue", 5900)

# 使用方法
folder = './1/reference-780'  # ← 替换为你自己的图像文件夹路径
is256 = False    # 是否要 resize 为 256*256
compute_channel_variance_per_image(folder, IsOrNot = is256)
