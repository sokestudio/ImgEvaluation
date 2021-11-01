import numpy as np
import math
import cv2
import os
from skimage.measure import compare_ssim
import skimage.measure
# 峰值信噪比
def psnr(target, ref):
	#将图像格式转为float64
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    # 直接相减，求差值
    diff = ref_data - target_data
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    rmse = math.sqrt(np.mean(diff ** 2.))
    # 精度
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps
    psnr=20*math.log10(255.0/rmse)
    return psnr

# 结构相似性
def ssim(imageA, imageB):
    # 为确保图像能被转为灰度图
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)

    # 通道分离，注意顺序BGR不是RGB
    (B1, G1, R1) = cv2.split(imageA)
    (B2, G2, R2) = cv2.split(imageB)

    # convert the images to grayscale BGR2GRAY
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 方法一
    #直接转换为灰度图，计算ssim
    (grayScore, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # print("gray SSIM: {}".format(grayScore))

    # 方法二
    #分离每个通道，计算ssim，再求平均值
    (score0, diffB) = compare_ssim(B1, B2, full=True)
    (score1, diffG) = compare_ssim(G1, G2, full=True)
    (score2, diffR) = compare_ssim(R1, R2, full=True)
    aveScore = (score0 + score1 + score2) / 3
    # print("BGR average SSIM: {}".format(aveScore))

    return  aveScore
#
##数据集评估####
'''
original_path=r"C:\study_he\exp_result\LOL15\high"
contrast_path=r"C:\study_he\RetinexNet-master\Retinex_n"
psnr_all=0
ssim_all=0
rmse_all=0
nrmse_all=0
entropy_all=0
length=len(os.listdir(original_path))
for file in os.listdir(original_path):
    original_img=os.path.join(original_path,file)
    contrast_img=os.path.join(contrast_path,file)
    original=cv2.imread(original_img)
    contrast = cv2.imread(contrast_img, 1)
    # 峰值信噪比
    psnrValue = psnr(original, contrast)
    # 结构相似性
    ssimValue = ssim(original, contrast)
    # 均方误差
    mse = skimage.measure.compare_mse(original, contrast)
    # 均方根误差
    rmse = math.sqrt(mse)
    # 归一化均方根误差
    nrmse = skimage.measure.compare_nrmse(original, contrast, norm_type='Euclidean')
    # 信息熵
    entropy = skimage.measure.shannon_entropy(contrast, base=2)
    psnr_all+=psnrValue
    ssim_all+=ssimValue
    rmse_all+=rmse
    nrmse_all+=nrmse
    entropy_all+=entropy

psnr_avg=psnr_all/length
ssim_avg=ssim_all/length
rmse_avg=rmse_all/length
nrmse_avg=nrmse_all/length
entropy_avg=entropy_all/length
print(rmse_avg)
print(nrmse_avg)
print(ssim_avg)
print(psnr_avg)
print(entropy_avg)

'''
#单张图像预测
original = cv2.imread(r"C:\Users\pc\Desktop\Data20210812_2\DefectsImg_Cam02_1.bmp")      # numpy.adarray
contrast = cv2.imread(r"C:\Users\pc\Desktop\Data20210812_2\DefectsImg_Cam02_2.bmp",1)
#峰值信噪比
psnrValue = psnr(original,contrast)
#结构相似性
ssimValue = ssim(original,contrast)
# 均方误差
mse = skimage.measure.compare_mse(original, contrast)
# 均方根误差
rmse = math.sqrt(mse)
# 归一化均方根误差
nrmse = skimage.measure.compare_nrmse(original, contrast, norm_type='Euclidean')
# 信息熵
entropy = skimage.measure.shannon_entropy(contrast, base=2)
print(rmse)
print(nrmse)
print(ssimValue)
print(psnrValue)
print(entropy)
