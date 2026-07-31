## 主函数

from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Analysis_function import wpsnr
import cv2 as cv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from matplotlib import pyplot as plt
from Analysis_function import plt_hist, plt_corr, npcr, uaci, information_entropy, calculate_correlations, plaintext_sensitivity
from utils4 import encryption, dencryption, load_model
import skimage
import time
import warnings
warnings.filterwarnings("ignore")

############ 调试
im = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/avion.ppm',0))
# im = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/portofino.ppm',0))
cover = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/sailboat.ppm',0))

# im = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/baboon.ppm'))
# cover = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/lgthouse.ppm'))

# im = cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/elephant.ppm')
# im = np.double(cv.resize(im, [256, 256]))
# cover = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/beeflowr.ppm',0))

# im = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/peppers.ppm',0))
# cover = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/house.ppm'))

# im = cv.imread('E:\date\ccia_CVG_image\color_image_512/lena.ppm')
# cover = cv.imread('E:\date\ccia_CVG_image\color_image_512/peppers.ppm')
# im = cv.imread('E:\date\ccia_CVG_image\color_image_512/baboon.ppm',0)
# im = np.double(cv.resize(im, [256, 256]))
# cover = np.double(cv.resize(cover, [256, 256]))


sec_key = '8d5ab8ba5340fce4420829ad5d12a0e45dacb0858544163d04c1d02b73e3697d'
hrp = 3  #

# emb_type = 'spa'
# dco = [4,4,4,4]
emb_type = 'haar'
dco = [2,4,4,8]

im_shape = im.shape
m, n, k = im.shape if im.ndim == 3 else (*im.shape, 1)
if k == 1:
    module = load_model('model2/model_epoch_gray.pth', in_channels=1, device='cpu')
else:
    module = load_model('model2/model_epoch_color.pth', in_channels=3, device='cpu')

cip, nc, dynamic_key, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
rim, nc2 = dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)

print(PSNR(np.uint8(cip),np.uint8(cover)))
print(SSIM(np.uint8(cip),np.uint8(cover),multichannel=True))
print(PSNR(np.uint8(im),np.uint8(rim)))
print(SSIM(np.uint8(im),np.uint8(rim),multichannel=True))

# # portofino
# # 在 im 上绘制红色框
# im_with_box = im.copy()  # 复制图像，避免修改原图
# cv.rectangle(im_with_box, (330, 330), (430, 430), 255, 3)  # 255 白色
# # 在 im 上绘制红色框
# rim_with_box = rim.copy()  # 复制图像，避免修改原图
# cv.rectangle(rim_with_box, (330, 330), (430, 430), 255, 3)  # 255 白色


# # 在 im 上绘制红色框
# im_with_box = im.copy()  # 复制图像，避免修改原图
# cv.rectangle(im_with_box, (70, 150), (170, 250), 255, 3)  # 255 白色
# # 在 im 上绘制红色框
# rim_with_box = rim.copy()  # 复制图像，避免修改原图
# cv.rectangle(rim_with_box, (70, 150), (170, 250), 255, 3)  # 255 白色

cv.imshow('im', np.uint8(im))
cv.imshow('cover', np.uint8(cover))
cv.imshow('nc', np.uint8(nc))
cv.imshow('cip', np.uint8(cip))
cv.imshow('cover-cip', np.uint8(np.clip(abs(cover - cip) * 50,0,256)))
cv.imshow('cover-cip-2', np.uint8(255 - abs(cover - cip) * 50))
# cv.imshow('rim', np.uint8(rim_with_box))
# cv.imshow('rim2', np.uint8(rim[330:430,330:430]))
cv.imshow('im-rim', np.uint8(np.clip(abs(im - rim) * 30,0,256)))
plt.show()
cv.waitKey(0)



# ############ 传统性能分析
#
# if __name__ == '__main__':
#     "常规"
#
#     im = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/baboon.ppm'))
#     cover = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/lgthouse.ppm'))
#
#     sec_key = '8d5ab8ba5340fce4420829ad5d12a0e45dacb0858544163d04c1d02b73e3697d'
#     module = load_model('model2/model_epoch_color.pth', in_channels=3, device='cpu')
#     dco = [4,4,4,4]
#     hrp = 3
#     emb_type = 'spa'
#     im_shape = im.shape
#     time1 = time.time()
#     cip, nc, dynamic_key, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
#     time2 = time.time()
#     rim, nc2 = dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
#
#     im1 = np.copy(im)
#     im1[0, 0, :] = im1[0, 0, :] + 1
#     cip2, nc2, dynamic_key2, max_min2, index_x12 = encryption(im1, cover, sec_key, hrp, emb_type, module, dco)
#
#     print('加密时间:', time2 - time1)
#     print('信息熵：', information_entropy(np.uint8(nc)), information_entropy(np.uint8(cip)),
#           information_entropy(np.uint8(rim)))
#     print('类噪声相关性：', calculate_correlations(np.uint8(nc)))
#     print('密文相关性：', calculate_correlations(np.uint8(cip)))
#     print('解密图像相关性：', calculate_correlations(np.uint8(rim)))
#
#     print('明文敏感性', plaintext_sensitivity(nc, nc2), plaintext_sensitivity(cip, cip2))
#
#     cv.imshow('im', np.uint8(im))
#     cv.imshow('cover', np.uint8(cover))
#     cv.imshow('tcip', np.uint8(nc))
#     cv.imshow('cip', np.uint8(cip))
#     cv.imshow('cip-cover', np.uint8(50 * np.abs(cip - cover)))
#     cv.imshow('rim', np.uint8(rim))
#     cv.imshow('im-rim', np.uint8(30 * np.abs(im - rim)))
#     plt.show()
#     cv.waitKey(0)

    # # # 密钥敏感性
    # key1=np.copy(dynamic_key)
    # key1[0]=key1[0] + 1*10**-14
    # rim1, nc2 = dencryption(cip, cover, key1, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    #
    # key2 = np.copy(dynamic_key)
    # key2[1][0] = key2[1][0] + 1*10 ** -14
    # rim2, nc2 = dencryption(cip, cover, key2, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    #
    # key3 = np.copy(dynamic_key)
    # key3[1][1] = key3[1][1] + 1*10 ** -14
    # rim3, nc2 = dencryption(cip, cover, key3, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    #
    # key4 = np.copy(dynamic_key)
    # key4[1][2] = key4[1][2] + 1*10 ** -14
    # rim4, nc2 = dencryption(cip, cover, key4, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    #
    # key5 = np.copy(dynamic_key)
    # key5[1][3] = key5[1][3] + 1*10 ** -14
    # rim5, nc2 = dencryption(cip, cover, key5, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    #
    # cv.imshow('rim1', np.uint8(rim1))
    # cv.imshow('rim2', np.uint8(rim2))
    # cv.imshow('rim3', np.uint8(rim3))
    # cv.imshow('rim4', np.uint8(rim4))
    # cv.imshow('rim5', np.uint8(rim5))
    # plt.show()
    # cv.waitKey(0)

    #统计特性
    # plt_hist(np.uint8(im), 11,(5,4),14)
    # plt_corr(np.uint8(im), 12,(4,4),14)
    # plt_hist(np.uint8(cover), 21,(5,4),14)
    # plt_corr(np.uint8(cover), 22,(4,4),14)
    # plt_hist(np.uint8(nc), 31,(5,4),14)
    # plt_corr(np.uint8(nc), 32,(4,4),14)
    # plt_hist(np.uint8(cip), 41,(5,4),14)
    # plt_corr(np.uint8(cip), 42,(4,4),14)
    # plt_hist(np.uint8(rim), 51, (5, 4), 14)
    # plt_corr(np.uint8(rim), 52, (4, 4), 14)
    # plt.show()
    # cv.waitKey(0)


    # # #明文敏感性
    # im1 = np.copy(im)
    # im1[1,1]=im1[1,1]+1
    # cip1, nc1, dynamic_key, max_min, index_x1 = encryption(im1, cover, sec_key, hrp, emb_type, module, dco)
    # im2 = np.copy(im)
    # im2[256, 128] = im2[256, 128] + 1
    # cip2, nc2, dynamic_key, max_min, index_x1 = encryption(im2, cover, sec_key, hrp, emb_type, module, dco)
    # im3 = np.copy(im)
    # im3[256, 128] = im[1,1]
    # im3[1, 1] = im[256, 128]
    # cip3, nc3, dynamic_key, max_min, index_x1 = encryption(im3, cover, sec_key, hrp, emb_type, module, dco)
    # print("npcr_nc1:",npcr(nc,nc1))
    # print("uaci_nc1:", uaci(nc, nc1))
    # print("ssim_nc1:", SSIM(np.uint8(nc),np.uint8(nc1),multichannel=True))
    # print("npcr_nc2:",npcr(nc,nc2))
    # print("uaci_nc2:", uaci(nc, nc2))
    # print("ssim_nc2:", SSIM(np.uint8(nc),np.uint8(nc2),multichannel=True))
    # print("npcr_nc3:",npcr(nc,nc3))
    # print("uaci_nc3:", uaci(nc, nc3))
    # print("ssim_nc3:", SSIM(np.uint8(nc),np.uint8(nc3),multichannel=True))
    # print("npcr_cip1:",npcr(cip,cip1))
    # print("uaci_cip1:", uaci(cip,cip1))
    # print("ssim_cip1:", SSIM(np.uint8(cip),np.uint8(cip1),multichannel=True))
    # print("npcr_cip2:",npcr(cip,cip2))
    # print("uaci_cip2:", uaci(cip, cip2))
    # print("ssim_cip2:", SSIM(np.uint8(cip),np.uint8(cip2),multichannel=True))
    # print("npcr_cip3:",npcr(cip,cip3))
    # print("uaci_cip3:", uaci(cip, cip3))
    # print("ssim_cip3:", SSIM(np.uint8(cip),np.uint8(cip3),multichannel=True))

    # #选择明文攻击
    # print(dynamic_key)
    # im1 = np.copy(im)
    # im1[1, 1] = im1[1, 1] + 1
    # cip1, nc, dynamic_key1, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
    # rim1, nc2 = dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    # print(dynamic_key1)
    # cip2, nc, dynamic_key2, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
    # rim2, nc2 = dencryption(cip, cover, dynamic_key2, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    # print(dynamic_key2)
    # cv.imshow('im', np.uint8(im))
    # cv.imshow('cover', np.uint8(cover))
    # cv.imshow('im1', np.uint8(im1))
    # cv.imshow('im2', np.uint8(im))
    # cv.imshow('rim1', np.uint8(rim1))
    # cv.imshow('rim2', np.uint8(rim2))

    # # # 鲁棒性
    # im = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/baboon.ppm'))
    # cover = np.double(cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/lgthouse.ppm'))
    #
    # sec_key = '8d5ab8ba5340fce4420829ad5d12a0e45dacb0858544163d04c1d02b73e3697d'
    # module = load_model('model2/model_epoch_color.pth', in_channels=3, device='cpu')
    # dco = [4,4,4,4]
    # hrp = 3
    # emb_type = 'spa'
    # im_shape = im.shape
    # # # 椒盐
    # # cip, nc, dynamic_key, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
    # # # rim, nc2 = dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    # # os.makedirs('lu/sp', exist_ok=True)
    # #
    # # psnr_all = np.zeros(11)
    # # ssim_all = np.zeros(11)
    # #
    # # for i in range(11):
    # #     noise_density = 0.0005 * 4 * i
    # #     print(f"处理噪声密度: {noise_density:.4f}")
    # #     cip1 = np.round(skimage.util.random_noise(cip / 255, mode='s&p', amount=noise_density) * 255)
    # #     rim2, nc2 = dencryption(cip1, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    # #     psnr_i = PSNR(np.uint8(im),np.uint8(rim2))
    # #     ssim_i = SSIM(np.uint8(im),np.uint8(rim2),multichannel=True)
    # #     psnr_all[i] = psnr_i
    # #     ssim_all[i] = ssim_i
    # #     # 保存图像
    # #     save_rim1_path = f'lu/sp/rim1_noise_{i}.png'
    # #     cv.imwrite(save_rim1_path,np.uint8(rim2))
    # #
    # #     save_cip1_path = f'lu/sp/cip1_noise_{i}.png'
    # #     cv.imwrite(save_cip1_path,np.uint8(cip1))
    # # np.save('lu/sp/psnr.npy', psnr_all)
    # # np.save('lu/sp/ssim.npy', ssim_all)
    #
    # # psnr_all = np.load('lu/sp/psnr.npy')
    # # ssim_all = np.load('lu/sp/ssim.npy')
    # # xlable = np.arange(11) * 0.0005 * 4  # 绘制噪声测试结果
    # # fig, ax1 = plt.subplots(figsize=(7, 4))
    # #
    # # color1 = [0, 0.4470, 0.7410]  # 左Y轴 - PSNR
    # # ax1.set_xlabel('S&P noise density', fontsize=14, fontname='Times New Roman')
    # # ax1.set_ylabel('PSNR (dB)', fontsize=14, fontname='Times New Roman', color=color1)
    # # h1 = ax1.plot(xlable, psnr_all, '-o', linewidth=2, markersize=6,
    # #               markerfacecolor='b', color=color1, label='PSNR')
    # # ax1.tick_params(axis='y', labelcolor=color1)
    # # ax1.set_ylim([min(psnr_all) - 1, max(psnr_all) + 1])
    # #
    # # ax2 = ax1.twinx()  # 右Y轴 - SSIM
    # # color2 = [0.8500, 0.3250, 0.0980]
    # # ax2.set_ylabel('SSIM', fontsize=14, fontname='Times New Roman', color=color2)
    # # h2 = ax2.plot(xlable, ssim_all, '-s', linewidth=2, markersize=6,
    # #               markerfacecolor='r', color=color2, label='SSIM')
    # # ax2.tick_params(axis='y', labelcolor=color2)
    # # ax2.set_ylim([min(ssim_all) - 0.01, max(ssim_all) + 0.05])
    # #
    # # ax1.tick_params(axis='both', labelsize=12)  # 设置字体
    # # ax2.tick_params(axis='both', labelsize=12)
    # #
    # # lines = h1 + h2  # 添加网格和图例
    # # labels = [l.get_label() for l in lines]
    # # ax1.legend(lines, labels, loc='lower left', fontsize=12)
    # # plt.tight_layout()
    # # plt.show()
    #
    # #剪切
    # cip, nc, dynamic_key, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
    # # rim, nc2 = dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    # os.makedirs('lu/cut', exist_ok=True)
    #
    # psnr_all = np.zeros(11)
    # ssim_all = np.zeros(11)
    #
    # for i in range(11):
    #     cut_size = 8 * i
    #     print(f"剪切大小: {cut_size:.4f}")
    #     cip1 = cip.copy()
    #     if cut_size > 0:
    #         cip1[:cut_size,:cut_size,:] = 0
    #     rim2, nc2 = dencryption(cip1, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    #     psnr_i = PSNR(np.uint8(im),np.uint8(rim2))
    #     ssim_i = SSIM(np.uint8(im),np.uint8(rim2),multichannel=True)
    #     psnr_all[i] = psnr_i
    #     ssim_all[i] = ssim_i
    #     # 保存图像
    #     save_rim1_path = f'lu/cut/rim1_noise_{i}.png'
    #     cv.imwrite(save_rim1_path,np.uint8(rim2))
    #
    #     save_cip1_path = f'lu/cut/cip1_noise_{i}.png'
    #     cv.imwrite(save_cip1_path,np.uint8(cip1))
    #
    # np.save('lu/cut/psnr.npy', psnr_all)
    # np.save('lu/cut/ssim.npy', ssim_all)
    # # 绘制噪声测试结果
    # psnr_all = np.load('lu/cut/psnr.npy')
    # ssim_all = np.load('lu/cut/ssim.npy')
    # xlable = np.arange(11) * 8
    # fig, ax1 = plt.subplots(figsize=(7, 4))
    #
    # # 左Y轴 - PSNR
    # color1 = [0, 0.4470, 0.7410]
    # ax1.set_xlabel('Cut size', fontsize=14, fontname='Times New Roman')
    # ax1.set_ylabel('PSNR (dB)', fontsize=14, fontname='Times New Roman', color=color1)
    # h1 = ax1.plot(xlable, psnr_all, '-o', linewidth=2, markersize=6,
    #               markerfacecolor='b', color=color1, label='PSNR')
    # ax1.tick_params(axis='y', labelcolor=color1)
    # ax1.set_ylim([min(psnr_all) - 0.5, max(psnr_all) + 0.5])
    #
    # # 右Y轴 - SSIM
    # ax2 = ax1.twinx()
    # color2 = [0.8500, 0.3250, 0.0980]
    # ax2.set_ylabel('SSIM', fontsize=14, fontname='Times New Roman', color=color2)
    # h2 = ax2.plot(xlable, ssim_all, '-s', linewidth=2, markersize=6,
    #               markerfacecolor='r', color=color2, label='SSIM')
    # ax2.tick_params(axis='y', labelcolor=color2)
    # ax2.set_ylim([min(ssim_all) - 0.01, max(ssim_all) + 0.01])
    #
    # # 设置字体
    # ax1.tick_params(axis='both', labelsize=12)
    # ax2.tick_params(axis='both', labelsize=12)
    #
    # # 添加网格和图例
    # lines = h1 + h2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper right', fontsize=12)
    # plt.tight_layout()
    # plt.show()



# ############ 加密文件夹中图片并保持
# if __name__ == '__main__':
#     # flies = 'cover_1024'
#     flies = 'cover_512'
#     # flies = 'cover_256'
#     images_name = sorted(os.listdir(flies))
#     num_image = len(images_name)
#     r1 = np.random.randint(num_image, size=num_image)
#
#     sec_key = '8d5ab8ba5340fce4420829ad5d12a0e45dacb0858544163d04c1d02b73e3697d'
#     module = load_model('model2/model_epoch_gray.pth', in_channels=1, device='cpu')
#     hrp = 3  #
#     # emb_type = 'spa'
#     # dco = [4,4,4,4]
#     emb_type = 'haar'
#     dco = [2, 4, 4, 8]
#
#     path = 'cip/'+ flies + '/' + emb_type
#
#     psnr_all = []
#     ssim_all = []
#
#     os.makedirs(path, exist_ok=True)
#
#     for i in range(num_image):
#         # print(i)
#         if i < num_image - 1:
#             im = np.double(cv.imread(flies + '/' + images_name[i + 1], 0))
#         else:
#             im = np.double(cv.imread(flies + '/' + images_name[0], 0))
#         cover = np.double(cv.imread(flies + '/' + images_name[i], 0))
#
#         im_shape = im.shape
#         cip, nc, dynamic_key, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
#
#         psnr_all.append(PSNR(np.uint8(cip),np.uint8(cover)))
#         ssim_all.append(SSIM(np.uint8(cip),np.uint8(cover)))
#
#         cv.imwrite(path+'/'+images_name[i],np.uint8(cip))
#
#     psnr_all = np.array(psnr_all)
#     ssim_all = np.array(ssim_all)
#     print(np.mean(psnr_all), np.mean(ssim_all))
#
#     rim, nc2 = dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
#     cv.imshow('im', np.uint8(im))
#     cv.imshow('cover', np.uint8(cover))
#     cv.imshow('cip', np.uint8(cip))
#     cv.imshow('cover-cip', np.uint8(abs(cover - cip) * 50))
#     cv.imshow('cover-cip-2', np.uint8(255 - abs(cover - cip) * 50))
#     cv.imshow('rim', np.uint8(rim))
#     plt.show()
#     cv.waitKey(0)


############ 解密质量
# if __name__ == '__main__':
#     flies = 'cover_1024'
#     # flies = 'cover_512'
#     # flies = 'cover_256'
#     images_name = sorted(os.listdir(flies))
#     num_image = len(images_name)
#     r1 = np.random.randint(num_image, size=num_image)
#
#     sec_key = '8d5ab8ba5340fce4420829ad5d12a0e45dacb0858544163d04c1d02b73e3697d'
#     module = load_model('model2/model_epoch_gray.pth', in_channels=1, device='cpu')
#     hrp = 3  #
#     emb_type = 'spa'
#     dco = [4,4,4,4]
#     # emb_type = 'haar'
#     # dco = [2, 4, 4, 8]
#
#     path_rim = 'rim/'+ flies + '/' + 'rim'
#     path_res = 'rim/' + flies + '/' + 'res'
#
#     psnr_all = []
#     ssim_all = []
#
#     os.makedirs(path_rim, exist_ok=True)
#     os.makedirs(path_res, exist_ok=True)
#
#     for i in range(num_image):
#         # print(i)
#         if i < num_image - 1:
#             im = np.double(cv.imread(flies + '/' + images_name[i + 1], 0))
#         else:
#             im = np.double(cv.imread(flies + '/' + images_name[0], 0))
#         cover = np.double(cv.imread(flies + '/' + images_name[i], 0))
#
#         im_shape = im.shape
#         cip, nc, dynamic_key, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
#         rim, nc2 = dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
#
#         psnr_i = PSNR(np.uint8(rim), np.uint8(im))
#         ssim_i = SSIM(np.uint8(rim), np.uint8(im), multichannel=True)
#         psnr_all.append(psnr_i)
#         ssim_all.append(ssim_i)
#
#         name, ext = os.path.splitext(images_name[i])
#         cip_name = f"{name}_{psnr_i:.4f}_{ssim_i:.4f}{'.bmp'}"
#         cv.imwrite(path_rim + '/' + cip_name, np.uint8(rim))
#         cv.imwrite(path_res + '/' + cip_name, np.uint8(abs(rim - im) * 50))
#
#     psnr_all = np.array(psnr_all)
#     ssim_all = np.array(ssim_all)
#     print(np.mean(psnr_all), np.mean(ssim_all))
#
#     cv.imshow('im', np.uint8(im))
#     cv.imshow('cover', np.uint8(cover))
#     cv.imshow('cip', np.uint8(cip))
#     cv.imshow('cover-cip', np.uint8(abs(cover - cip) * 50))
#     cv.imshow('cover-cip-2', np.uint8(255 - abs(cover - cip) * 50))
#     cv.imshow('rim', np.uint8(rim))
#     plt.show()
#     cv.waitKey(0)




