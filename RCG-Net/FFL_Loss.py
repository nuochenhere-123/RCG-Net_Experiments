import tensorflow as tf
import skimage.io as io
import numpy as np

def fft2d_with_normalization(x):
    # 计算 2D FFT
    freq = tf.signal.fft2d(tf.cast(x, tf.complex64))
    
    # 获取形状
    shape = tf.shape(x)
    height = tf.cast(shape[-2], tf.float32)
    width = tf.cast(shape[-1], tf.float32)
    
    # 正交归一化
    normalization_factor = tf.sqrt(height * width)
    # 将归一化因子转换为复数类型
    normalization_factor = tf.cast(normalization_factor, tf.complex64)
    
    freq_normalized = freq / normalization_factor
    
    return freq_normalized

    
class FocalFrequencyLoss(tf.keras.losses.Loss):
    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False, name="focal_frequency_loss"):
        super(FocalFrequencyLoss, self).__init__(name=name)
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        
    def tensor2freq(self, x):
        # 分割图像块
        patch_factor = self.patch_factor
        _, h, w, _ = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, 'Patch factor应该可以整除图像的高和宽'
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w, :])

        # 堆叠成块张量，沿指定维度（在此处是1），堆叠后(N, num_patches, patch_h, patch_w, C)
        y = tf.stack(patch_list, axis=1)
        # print(y)
        # 执行2D FFT（实数到复数，标准化）
        # freq = tf.signal.fft2d(tf.cast(y, tf.complex64))
        freq = fft2d_with_normalization(y)
        freq_real = tf.math.real(freq)
        freq_imag = tf.math.imag(freq)
        # print(freq_imag) # (4, 1, 64, 64, 3)
        freq = tf.stack([freq_real, freq_imag], axis=-1)
        # print(freq) # (4, 1, 3, 64, 64, 2)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # 频谱权重矩阵
        if matrix is not None:
            weight_matrix = tf.stop_gradient(matrix)
        else:
            # 如果矩阵是在线计算的：连续的，动态的，基于当前欧几里得距离
            matrix_tmp = (recon_freq - real_freq) ** 2
            # print(matrix_tmp, "matrix_tmp")
            matrix_tmp = tf.pow(tf.math.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]), self.alpha)
            # print(matrix_tmp, "matrix_tmp")
            
            # 是否使用对数调整频谱权重矩阵
            if self.log_matrix:
                matrix_tmp = tf.math.log(matrix_tmp + 1.0)
            # print(matrix_tmp, "matrix_tmp") # (4, 1, 3, 64, 64)
            
            # 是否使用基于批次的统计数据来计算频谱权重矩阵
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / tf.reduce_max(matrix_tmp)
            else:
                matrix_tmp = matrix_tmp / tf.reduce_max(tf.reduce_max(matrix_tmp, axis=-1, keepdims=True), axis=-2, keepdims=True)
            # print(matrix_tmp)
            # print(matrix_tmp.shape) # (4, 1, 64, 64, 3)
            matrix_tmp = tf.where(tf.math.is_nan(matrix_tmp), tf.zeros_like(matrix_tmp), matrix_tmp)
            weight_matrix = tf.clip_by_value(matrix_tmp, clip_value_min=0.0, clip_value_max=1.0)

        # 确认权重矩阵的值在[0, 1]范围内
        tf.debugging.assert_greater_equal(tf.reduce_min(weight_matrix), 0.0, '频谱权重矩阵的值应该大于或等于0')
        tf.debugging.assert_less_equal(tf.reduce_max(weight_matrix), 1.0, '频谱权重矩阵的值应该小于或等于1')
    
        # 频率距离的欧几里得距离计算
        tmp = (recon_freq - real_freq) ** 2
        # print(recon_freq)
        # print(tmp, tmp.shape) # (4, 1, 3, 64, 64, 2)
        freq_distance = tmp[..., 0] + tmp[..., 1]
        # print(freq_distance)
        
        # 动态频谱加权（Hadamard积）
        loss = weight_matrix * freq_distance
        # print(tf.reduce_mean(loss))
        return tf.reduce_mean(loss)

    def call(self,y_pred, y_true, matrix=None):
        """计算Focal Frequency Loss."""
        pred_freq = self.tensor2freq(y_pred)
        target_freq = self.tensor2freq(y_true)
        # print("Predicted frequency (TensorFlow): ", pred_freq) # (4, 1, 64, 64, 3)
        # print("Target frequency (TensorFlow): ", target_freq) # (4, 1, 64, 64, 3)
        
        # 是否使用小批次平均频谱
        if self.ave_spectrum:
            pred_freq = tf.reduce_mean(pred_freq, axis=0, keepdims=True)
            target_freq = tf.reduce_mean(target_freq, axis=0, keepdims=True)
           

        # 计算Focal Frequency Loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

