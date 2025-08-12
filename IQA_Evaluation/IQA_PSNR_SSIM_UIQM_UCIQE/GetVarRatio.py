# 分别获取 两个文件夹内所有同名图像（增强后/增强前）的 RGB 三通道各自的方差和三通道方差均值 增强后/增强前 的 直方图分布
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def compute_variance_change_ratio(folder_raw, folder_gt):
    # 图像文件名交集
    raw_files = {f for f in os.listdir(folder_raw) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
    gt_files = {f for f in os.listdir(folder_gt) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
    common_files = sorted(list(raw_files & gt_files))

    if not common_files:
        print("两个文件夹没有同名图像！")
        return

    ratios_R, ratios_G, ratios_B, ratios_Mean = [], [], [], []

    for fname in common_files:
        raw_path = os.path.join(folder_raw, fname)
        gt_path = os.path.join(folder_gt, fname)

        raw = cv2.imread(raw_path)
        gt = cv2.imread(gt_path)
        if raw is None or gt is None:
            print(f"跳过无法读取的图像: {fname}")
            continue

        var_raw_b = np.var(raw[:, :, 0])
        var_raw_g = np.var(raw[:, :, 1])
        var_raw_r = np.var(raw[:, :, 2])
        mean_raw = (var_raw_r + var_raw_g + var_raw_b) / 3

        var_gt_b = np.var(gt[:, :, 0])
        var_gt_g = np.var(gt[:, :, 1])
        var_gt_r = np.var(gt[:, :, 2])
        mean_gt = (var_gt_r + var_gt_g + var_gt_b) / 3

        # 防止除以0，加入极小值
        eps = 1e-6
        ratios_R.append((fname, var_gt_r / (var_raw_r + eps)))
        ratios_G.append((fname, var_gt_g / (var_raw_g + eps)))
        ratios_B.append((fname, var_gt_b / (var_raw_b + eps)))
        ratios_Mean.append((fname, mean_gt / (mean_raw + eps)))

    # 设置字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    folder_name = os.path.basename(os.path.normpath(folder_raw))

    def analyze_and_plot(ratios, channel_name, color):
        values = [v for _, v in ratios]

        # 输出最大三个
        print(f"\n【{channel_name} 方差比例 - 最大三个】")
        for name, val in sorted(ratios, key=lambda x: -x[1])[:3]:
            print(f"  {name}: {val:.2f}")

        # 输出最小三个
        print(f"【{channel_name} 方差比例 - 最小三个】")
        for name, val in sorted(ratios, key=lambda x: x[1])[:3]:
            print(f"  {name}: {val:.2f}")

        # 绘制直方图
        plt.figure(figsize=(8, 5))
        bins = np.linspace(0, min(5, max(values)), 30)  # 限制最大绘图范围为5
        plt.hist(values, bins=bins, edgecolor='black', color=color)
        plt.title(f"{folder_name} 中 {channel_name}通道方差 增强后/增强前的比例 分布")
        plt.xlabel("增强后 / 增强前 的方差比例")
        plt.ylabel("图像数量")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_name = f"var-{folder_name}-{channel_name}-changeRatio.png"
        plt.savefig(save_name)
        print(f"  ⬆ 直方图已保存为 {save_name}")

    # 分析并绘制每个通道及均值
    analyze_and_plot(ratios_R, "R", "salmon")
    analyze_and_plot(ratios_G, "G", "mediumseagreen")
    analyze_and_plot(ratios_B, "B", "cornflowerblue")
    analyze_and_plot(ratios_Mean, "Mean", "orchid")

# 示例调用
raw_folder = "./1/raw-780"
gt_folder = "./1/reference-780"
compute_variance_change_ratio(raw_folder, gt_folder)
