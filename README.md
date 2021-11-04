# ImgEvaluation
全参考图像质量评价

## 相关库安装
- pip install numpy
- pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
- pip install scikit-image==0.15.0 -U -i https://pypi.tuna.tsinghua.edu.cn/simple

## 指标
### 1.峰值信噪比 PSNR
  峰值信噪比，一种评价图像的客观标准，用来评估图像的保真性。峰值信噪比经常用作图像压缩等领域中信号重建质量的测量方法，它常简单地通过均方差（MSE）进行定义，使用两个m×n单色图像I和K。PSNR的单位为分贝dB。
  其中，MAXI是表示图像点颜色的最大数值，如果每个采样点用 8 位表示，那么就是 255。PSNR值越大，就代表失真越少，图像压缩中典型的峰值信噪比值在 30 到 40dB 之间，小于30dB时考虑图像无法忍受。
### 2.结构相似性 SSIM
  结构相似性，是一种衡量两幅图像相似度的指标，也是一种全参考的图像质量评价指标，它分别从亮度、对比度、结构三方面度量图像相似性。
### 3.均方差 RMSE

### 4.归一化均方根误差 NRMSE

### 5.信息熵 ENTROPY
