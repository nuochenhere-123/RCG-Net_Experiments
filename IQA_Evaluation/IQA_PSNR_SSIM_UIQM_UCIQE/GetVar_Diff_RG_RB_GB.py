# 获取 某个文件夹内所有图像的 RGB 三通道方差两两之差 的 直方图分布
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def compute_channel_variance_per_image(image_folder, IsOrNot=False):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    all_var_r, all_var_g, all_var_b = [], [], []
    var_mean_list = []
    mean_var_per_image = []
    rg_diff_list = []
    rb_diff_list = []
    gb_diff_list = []

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

        rg_diff = abs(var_r - var_g)
        rb_diff = abs(var_r - var_b)
        gb_diff = abs(var_g - var_b)

        rg_diff_list.append((image_name, rg_diff))
        rb_diff_list.append((image_name, rb_diff))
        gb_diff_list.append((image_name, gb_diff))

        print(f"{image_name}: R_var={var_r:.2f}, G_var={var_g:.2f}, B_var={var_b:.2f}, "
              f"Mean_var={var_mean:.2f}, |R-G|={rg_diff:.2f}, |R-B|={rb_diff:.2f}, |G-B|={gb_diff:.2f}")

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

    sorted_mean_var = sorted(mean_var_per_image, key=lambda x: x[1])
    print(f"\n通道方差均值的全局平均: {global_mean:.2f}\n  最大三值：")
    for n, v in sorted(sorted_mean_var[-3:], key=lambda x: -x[1]):
        print(f"    {n}: {v:.2f}")
    print("  最小三值：")
    for n, v in sorted_mean_var[:3]:
        print(f"    {n}: {v:.2f}")

    # 设置支持中文的字体（如 SimHei 或 Microsoft YaHei）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    name_now = os.path.basename(os.path.normpath(image_folder))

    # 方差均值分布图
    plt.figure(figsize=(8, 5))
    bins = np.arange(0, max(var_mean_list) + 200, 200)   # 指定 每个 bin 的宽度都是 200
    plt.hist(var_mean_list, bins=bins, edgecolor='black', color='skyblue')
    below_5000 = sum(v < 5000 for v in var_mean_list)
    percentage = (below_5000 / len(var_mean_list)) * 100
    title_suffix = "-256" if IsOrNot else ""
    plt.title(f"{name_now}{title_suffix} 中 单张图像 的 三通道方差均值分布")
    plt.xlabel(f"方差  通道方差均值 < 5000 占比：{percentage:.2f}% ")
    plt.ylabel("图像数量")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"var-{name_now}{title_suffix}.png")
    print(f"\n方差均值分布图已保存为 var-{name_now}{title_suffix}.png")

    # 通用函数：绘制差值分布图并输出最大最小
    def plot_diff_hist(diff_list, label, color):
        values = [x[1] for x in diff_list]
        plt.figure(figsize=(8, 5))
        bins1 = np.arange(0, max(values) + 300, 300)   # 指定 每个 bin 的宽度都是 200
        plt.hist(values, bins=bins1, edgecolor='black', color=color)   
        # plt.hist(values, bins=30, edgecolor='black', color=color)   # 指定要分成 30 个 bin（区间）
        plt.title(f"{name_now} 中 |{label}| 方差差值分布图")
        plt.xlabel("方差差值")
        plt.ylabel("图像数量")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"var-{name_now}-diff-{label.replace('-', '')}.png")
        print(f"\n|{label}| 方差差值分布图已保存。")
        print(f"\n|{label}| 最大三值：")
        for n, v in sorted(diff_list, key=lambda x: -x[1])[:3]:
            print(f"    {n}: {v:.2f}")
        print(f"|{label}| 最小三值：")
        for n, v in sorted(diff_list, key=lambda x: x[1])[:3]:
            print(f"    {n}: {v:.2f}")

    # 分别绘制差值分布图
    plot_diff_hist(rg_diff_list, "R-G", "lightskyblue")
    plot_diff_hist(rb_diff_list, "R-B", "lightgreen")
    plot_diff_hist(gb_diff_list, "G-B", "lightcoral")

# 用法
folder = './reference-780'  # ← 替换为你的图像文件夹路径
is256 = False
compute_channel_variance_per_image(folder, IsOrNot=is256)
