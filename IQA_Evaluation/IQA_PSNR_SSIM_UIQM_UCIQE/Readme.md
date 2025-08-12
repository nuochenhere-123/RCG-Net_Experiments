### 有参考图像测试（PSNR & SSIM & UIQM & UCIQE）：  
python **PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py** {Testingset} {reference}  
e.g: python PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py .\UWMamba-110-256 .\110_reference_256  
python PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py .\RetinexBased-110 .\UIEB-reference-890   

### 无参考图像(UIQM & UCIQE)：  
python python **UCIQE_UIQM_inLAB.py** {Testingset}  
e.g: python python UCIQE_UIQM_inLAB.py .\upgrade2_3-U45  
python UCIQE_UIQM_inLAB.py ./RetinexBased-C60

### 从图像中获取它的RGB三通道图像 —— Get_R_G_B_from_img.py


### 删除 某文件夹内 指定命名格式图片/删除不符合某命名格式的图片 —— delete.py

### 删除某文件夹内和另一个文件夹内重复的图片 —— deleteExist.py

### 删除 图像的宽度或高度小于某个阈值（256像素）的图片，并删除该图像 对应 reference 文件夹 中图片 —— ImageSizeChoose.py

### 其他未提及的 python 文件 的 用法 可在对应文件名或对应文件的第一行注释中找到


