"""
一些图像加密所用到的函数
输入图像的彩色图像必须为长宽高的形式
"""
import argparse
import math
import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import scipy.signal as signal
from scipy.fft import ifft2, fftshift
from scipy.linalg import lstsq


def npcr(a, b):
    # 像素变化率
    if a.shape == b.shape:

        if len(a.shape) == 3:
            [rows, columns, pages] = a.shape
            counter = 0
            for k in range(pages):
                for j in range(columns):
                    for i in range(rows):
                        if a[i, j, k] != b[i, j, k]:
                            counter += 1
            c = counter/(rows*columns*pages)
        else:
            [rows, columns] = a.shape
            counter = 0
            for j in range(columns):
                for i in range(rows):
                    if a[i, j] != b[i, j]:
                        counter += 1
            c = counter / (rows * columns)

    else:
        print('两图像必须大小相同')
        c = 0.0

    return c

def uaci(image1, image2):
    # 归一化变化强度
    if image1.shape == image2.shape:
        if len(image1.shape) == 3:
            [rows, columns, pages] = image1.shape
            counter = 0
            for k in range(pages):
                for j in range(columns):
                    for i in range(rows):
                        counter = (math.fabs(int(image1[i, j, k]) - int(image2[i, j, k])))/255 + counter
            c = counter / (rows * columns * pages)
        else:
            [rows, columns] = image1.shape
            counter = 0
            for j in range(columns):
                for i in range(rows):
                    counter = (math.fabs(int(image1[i, j]) - int(image2[i, j])))/255 + counter
            c = counter / (rows * columns)
    else:
        print('两图像必须大小相同')
        c = 0.0

    return c

def plaintext_sensitivity(im1,im2):
    NPCR = npcr(im1,im2)
    UACI = uaci(im1,im2)
    SSIM = ssim(np.uint8(im1),np.uint8(im2),multichannel=True)
    return np.round(NPCR,4), np.round(UACI,4), np.round(SSIM,4)


def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x = x.reshape(-1, )
    x_value_list = set([x[i] for i in range(x.shape[0])])
    # for i in range(x.shape[0])：
    #    x_value_list = set(x[i])

    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]   # x[x == x_value].shape[0] 求取x中等于x_value的个数
        logp = np.log2(p)
        ent -= p * logp

    return ent

def information_entropy(image):

    if len(image)==3:
        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]
        B_ENT = calc_ent(B)
        G_ENT = calc_ent(G)
        R_ENT = calc_ent(R)
        T1 = np.array([(B_ENT+G_ENT+R_ENT)/3,B_ENT, G_ENT, R_ENT])
    else:
        T1 = calc_ent(image)
    return np.round(T1,4)


def mssim(im1,im2,win):
    """
    :param im1: 第一张图像(输入为长宽高形式)
    :param im2: 第二张图像
    :param win: 窗口大小
    :return: 平均结构相似性
    """
    im1 = im1.astype(float)
    im2 = im2.astype(float)
    c1 = 2.55**2
    c2 = 7.65**2
    if len(im1.shape)==3:
        m,n,p=im1.shape
    else:
        im1 = np.expand_dims(im1,axis=2)
        im2 = np.expand_dims(im2,axis=2)
        m, n, p = im1.shape

    sz1 = int(math.floor(win[0] / 2))
    sz2 = int(math.floor(win[1] / 2))

    mer = []

    for i in range(sz1,m-sz1):
        for j in range(sz2,n-sz2):
            six = im1[i-sz1:i+sz1,j-sz2:j+sz2,:].ravel()
            siy = im2[i-sz1:i+sz1,j-sz2:j+sz2,:].ravel()
            meux = np.mean(six)
            meuy = np.mean(siy)
            sigx = np.std(six)
            sigy = np.std(siy)
            sigxy = np.mean((six-meux)*(siy-meuy))
            er = ((2*meux*meuy+c1)*(2*sigxy+c2))/((meux**2+meuy**2+c1)*(sigx**2+sigy**2+c2))
            mer.append(er)

    return sum(mer)/(len(mer))


def psnr(a,b):
    """
    两图的峰值信噪比
    """
    a = a.astype(float).reshape(-1, )
    b = b.astype(float).reshape(-1, )
    mse = np.sum((a-b)**2)
    psnr = 10*math.log10(255**2*a.shape[0]/mse)
    return psnr


# def plt_hist(img,a):
#     """
#     画出直方图
#     """
#     if len(img.shape) == 3:
#         plt.figure(a)
#         plt.hist(img[:, :, 0].ravel(), 256, [0, 256], color='blue')
#         plt.hist(img[:, :, 1].ravel(), 256, [0, 256], color='green')
#         plt.hist(img[:, :, 2].ravel(), 256, [0, 256], color='red')
#     else:
#         plt.figure(a)
#         plt.hist(img.ravel(), 256, [0, 256])
#
#
# def plt_corr(img,a):
#     """
#     画相关性
#     """
#     m = img.shape[0]
#     n = img.shape[1]
#     tem = np.ones((img.shape[0], img.shape[1]))
#     if len(img.shape) == 3:
#         fig = plt.figure(a)
#         ax = fig.gca(projection='3d')
#         ax.plot(tem[:, 0:n-1].ravel(), img[:, 0:n-1, 0].ravel(), img[:, 1:n, 0].ravel(), '.',
#                            markersize=0.25, color='blue')
#         ax.plot(2 * tem[0:m-1, :].ravel(), img[0:m-1, :, 0].ravel(), img[1:m, :, 0].ravel(), '.',
#                            markersize=0.25, color='blue')
#         ax.plot(3 * tem[0:m-1, 0:n-1].ravel(), img[0:m-1, 0:n-1, 0].ravel(),
#                            img[1:m, 1:n, 0].ravel(), '.',
#                            markersize=0.25, color='blue')
#
#         ax.plot(5 * tem[:, 0:n-1].ravel(), img[:, 0:n-1, 1].ravel(), img[:, 1:n, 1].ravel(), '.',
#                            markersize=0.25, color='green')
#         ax.plot(6 * tem[0:m-1, :].ravel(), img[0:m-1, :, 1].ravel(), img[1:m, :, 1].ravel(), '.',
#                            markersize=0.25, color='green')
#         ax.plot(7 * tem[0:m-1, 0:n-1].ravel(), img[0:m-1, 0:n-1, 1].ravel(),
#                            img[1:m, 1:n, 1].ravel(), '.',
#                            markersize=0.25, color='green')
#
#         ax.plot(9 * tem[:, 0:n-1].ravel(), img[:, 0:n-1, 2].ravel(), img[:, 1:n, 2].ravel(), '.',
#                            markersize=0.25, color='red')
#         ax.plot(10 * tem[0:m-1, :].ravel(), img[0:m-1, :, 2].ravel(), img[1:m, :, 2].ravel(), '.',
#                            markersize=0.25, color='red')
#         ax.plot(11 * tem[0:m-1, 0:n-1].ravel(), img[0:m-1, 0:n-1, 2].ravel(),
#                            img[1:m, 1:n, 2].ravel(), '.',
#                            markersize=0.25, color='red')
#
#         plt.xticks([1, 2, 3, 5, 6, 7, 9, 10 ,11],
#                    ['Horizontal', 'Vertical', 'Diagonal', 'Horizontal', 'Vertical', 'Diagonal', 'Horizontal', 'Vertical', 'Diagonal'],
#                    rotation=20)
#
#         fig = plt.figure(a+a)
#         ax = fig.gca(projection='3d')
#         ax.plot(tem.ravel(), img[:, :, 0].ravel(), img[:, :, 1].ravel(), '.',
#                 markersize=0.25, color='yellow')
#         ax.plot(2 * tem.ravel(), img[:, :, 0].ravel(), img[:, :, 2].ravel(), '.',
#                 markersize=0.25, color='magenta')
#         ax.plot(3 * tem.ravel(), img[:, :, 1].ravel(),img[:, :, 2].ravel(), '.',
#                 markersize=0.25, color='cyan')
#         plt.xticks([1, 2, 3],
#                    ['R_G', 'R_B', 'B_G'],rotation=20)
#
#     else:
#         fig = plt.figure(a)
#         ax = fig.gca(projection='3d')
#         ax.plot(tem[:, 0:n - 1].ravel(), img[:, 0:n - 1].ravel(), img[:, 1:n].ravel(), '.',
#                 markersize=0.25, color='blue')
#         ax.plot(2 * tem[0:m - 1, :].ravel(), img[0:m - 1, :].ravel(), img[1:m, :].ravel(), '.',
#                 markersize=0.25, color='blue')
#         ax.plot(3 * tem[0:m - 1, 0:n - 1].ravel(), img[0:m - 1, 0:n - 1].ravel(),
#                 img[1:m, 1:n].ravel(), '.',
#                 markersize=0.25, color='blue')
#         plt.xticks([1, 2, 3],
#                    ['Horizontal', 'Vertical', 'Diagonal'], rotation=20)


def plt_hist(img, a, figsize=(8, 6), fontsize=12):
    """
    Draw histogram
    Parameters:
    img: input image
    a: figure number
    figsize: figure size (width, height) in inches
    fontsize: font size for axis labels and ticks
    """
    if len(img.shape) == 3:
        plt.figure(a, figsize=figsize)
        plt.hist(img[:, :, 0].ravel(), 256, [0, 255], color='blue')
        plt.hist(img[:, :, 1].ravel(), 256, [0, 255], color='green')
        plt.hist(img[:, :, 2].ravel(), 256, [0, 255], color='red')

        # 设置x轴和y轴标签
        # plt.xlabel('Pixel Value', fontsize=fontsize, labelpad=10)
        # plt.ylabel('Frequency', fontsize=fontsize, labelpad=10)

        # 设置刻度字体大小
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # # 关键修改：调整y轴刻度标签与轴的距离，避免被截断
        # ax = plt.gca()
        # ax.yaxis.set_tick_params(pad=8)  # 增加y轴标签与轴的距离

        # # 添加图例和标题
        # plt.title('Histogram', fontsize=fontsize + 2)
        # plt.legend(['Blue', 'Green', 'Red'], fontsize=fontsize)

        # 调整图形边距，确保所有标签可见
        plt.subplots_adjust(left=0.2, right=0.98, bottom=0.12, top=0.98)

    else:
        plt.figure(a, figsize=figsize)
        plt.hist(img.ravel(), 256, [0, 255], color='gray', alpha=0.7)

        # 设置x轴和y轴标签
        # plt.xlabel('Pixel Value', fontsize=fontsize, labelpad=10)
        # plt.ylabel('Frequency', fontsize=fontsize, labelpad=10)

        # 设置刻度字体大小
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # # 关键修改：调整y轴刻度标签与轴的距离，避免被截断
        # ax = plt.gca()
        # ax.yaxis.set_tick_params(pad=8)

        # 添加标题
        # plt.title('Histogram', fontsize=fontsize + 2)

        # 调整图形边距，确保所有标签可见
        plt.subplots_adjust(left=0.2, right=0.98, bottom=0.12, top=0.98)


def plt_corr(img, a, figsize=(10, 8), fontsize=12):
    """
    画相关性
    Parameters:
    img: input image
    a: figure number
    figsize: figure size (width, height) in inches
    fontsize: font size for axis labels and ticks
    """
    m = img.shape[0]
    n = img.shape[1]
    tem = np.ones((img.shape[0], img.shape[1]))

    if len(img.shape) == 3:
        fig = plt.figure(a, figsize=figsize)
        ax = fig.gca(projection='3d')
        ax.plot(tem[:, 0:n - 1].ravel(), img[:, 0:n - 1, 0].ravel(), img[:, 1:n, 0].ravel(), '.',
                markersize=0.25, color='blue')
        ax.plot(2 * tem[0:m - 1, :].ravel(), img[0:m - 1, :, 0].ravel(), img[1:m, :, 0].ravel(), '.',
                markersize=0.25, color='blue')
        ax.plot(3 * tem[0:m - 1, 0:n - 1].ravel(), img[0:m - 1, 0:n - 1, 0].ravel(),
                img[1:m, 1:n, 0].ravel(), '.',
                markersize=0.25, color='blue')

        ax.plot(5 * tem[:, 0:n - 1].ravel(), img[:, 0:n - 1, 1].ravel(), img[:, 1:n, 1].ravel(), '.',
                markersize=0.25, color='green')
        ax.plot(6 * tem[0:m - 1, :].ravel(), img[0:m - 1, :, 1].ravel(), img[1:m, :, 1].ravel(), '.',
                markersize=0.25, color='green')
        ax.plot(7 * tem[0:m - 1, 0:n - 1].ravel(), img[0:m - 1, 0:n - 1, 1].ravel(),
                img[1:m, 1:n, 1].ravel(), '.',
                markersize=0.25, color='green')

        ax.plot(9 * tem[:, 0:n - 1].ravel(), img[:, 0:n - 1, 2].ravel(), img[:, 1:n, 2].ravel(), '.',
                markersize=0.25, color='red')
        ax.plot(10 * tem[0:m - 1, :].ravel(), img[0:m - 1, :, 2].ravel(), img[1:m, :, 2].ravel(), '.',
                markersize=0.25, color='red')
        ax.plot(11 * tem[0:m - 1, 0:n - 1].ravel(), img[0:m - 1, 0:n - 1, 2].ravel(),
                img[1:m, 1:n, 2].ravel(), '.',
                markersize=0.25, color='red')

        # 修改：使用ha='right'让标签右端对齐刻度，配合旋转角度，并减小labelpad
        plt.xticks([1, 2, 3, 5, 6, 7, 9, 10, 11],
                   ['Horizontal', 'Vertical', 'Diagonal', 'Horizontal', 'Vertical', 'Diagonal', 'Horizontal',
                    'Vertical', 'Diagonal'],
                   rotation=45, fontsize=fontsize - 1, ha='right')
        ax.xaxis.set_tick_params(pad=-6)

        # plt.yticks(fontsize=fontsize, ha='left')
        plt.yticks(fontsize=fontsize)
        ax.zaxis.set_tick_params(labelsize=fontsize)
        # ax.set_xlabel('Direction', fontsize=fontsize, labelpad=1)  # 减小到1
        # ax.set_ylabel('Pixel Value', fontsize=fontsize, labelpad=10)
        # ax.set_zlabel('Pixel Value', fontsize=fontsize, labelpad=10)

        fig = plt.figure(a + a, figsize=figsize)
        ax = fig.gca(projection='3d')
        ax.plot(tem.ravel(), img[:, :, 0].ravel(), img[:, :, 1].ravel(), '.',
                markersize=0.25, color='yellow')
        ax.plot(2 * tem.ravel(), img[:, :, 0].ravel(), img[:, :, 2].ravel(), '.',
                markersize=0.25, color='magenta')
        ax.plot(3 * tem.ravel(), img[:, :, 1].ravel(), img[:, :, 2].ravel(), '.',
                markersize=0.25, color='cyan')

        plt.xticks([1, 2, 3],
                   ['R_G', 'R_B', 'B_G'], rotation=0, fontsize=fontsize - 1, ha='center')
        plt.yticks(fontsize=fontsize)
        ax.zaxis.set_tick_params(labelsize=fontsize)
        # ax.set_xlabel('Channel Pair', fontsize=fontsize, labelpad=1)  # 减小到1
        # ax.set_ylabel('Pixel Value', fontsize=fontsize, labelpad=10)
        # ax.set_zlabel('Pixel Value', fontsize=fontsize, labelpad=10)

    else:
        fig = plt.figure(a, figsize=figsize)
        ax = fig.gca(projection='3d')
        ax.plot(tem[:, 0:n - 1].ravel(), img[:, 0:n - 1].ravel(), img[:, 1:n].ravel(), '.',
                markersize=0.25, color='blue')
        ax.plot(2 * tem[0:m - 1, :].ravel(), img[0:m - 1, :].ravel(), img[1:m, :].ravel(), '.',
                markersize=0.25, color='blue')
        ax.plot(3 * tem[0:m - 1, 0:n - 1].ravel(), img[0:m - 1, 0:n - 1].ravel(),
                img[1:m, 1:n].ravel(), '.',
                markersize=0.25, color='blue')

        plt.xticks([1, 2, 3],
                   ['Horizontal', 'Vertical', 'Diagonal'], rotation=20, fontsize=fontsize - 1, ha='right')
        plt.yticks(fontsize=fontsize)
        ax.zaxis.set_tick_params(labelsize=fontsize)
        # ax.set_xlabel('Direction', fontsize=fontsize, labelpad=1)  # 减小到1
        # ax.set_ylabel('Pixel Value', fontsize=fontsize, labelpad=10)
        # ax.set_zlabel('Pixel Value', fontsize=fontsize, labelpad=10)


def compute_correlation(x, y):
    """
    计算两个向量的相关系数

    参数:
        x, y - 相邻像素值向量
    返回:
        r - 相关系数
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    std_x = np.sqrt(np.sum((x - x_mean) ** 2))
    std_y = np.sqrt(np.sum((y - y_mean) ** 2))

    if std_x == 0 or std_y == 0:
        r = 0
    else:
        r = cov_xy / (std_x * std_y)

    return r


def calculate_correlations(img):
    """
    计算图像的相关性系数
    如果是彩色图像：返回6个相关性（水平、垂直、对角、R_G、R_B、G_B）
    如果是灰度图像：返回3个相关性（水平、垂直、对角）

    参数:
        img - 输入图像 (灰度图: H, W 或 彩色图: H, W, 3)
    返回:
        results - 包含相关性的字典
    """
    # 判断是否为彩色图像
    is_color = len(img.shape) == 3 and img.shape[2] == 3

    # 转换为浮点数
    img_float = img.astype(np.float64)

    results = {}

    if is_color:
        # 彩色图像：计算三个方向的整体相关性（所有通道一起计算）
        m, n, c = img_float.shape

        # 水平方向：所有通道一起
        h_data = img_float.reshape(-1, c)  # 先reshape为 (m*n, c)
        # 但是水平方向需要保持行结构
        x_h = img_float[:, 0:n - 1, :].reshape(-1, c)  # (m*(n-1), c)
        y_h = img_float[:, 1:n, :].reshape(-1, c)  # (m*(n-1), c)
        # 将所有通道展平为向量
        x_h_flat = x_h.flatten()
        y_h_flat = y_h.flatten()
        results['horizontal'] = np.round(compute_correlation(x_h_flat, y_h_flat),4)

        # 垂直方向
        x_v = img_float[0:m - 1, :, :].reshape(-1, c)
        y_v = img_float[1:m, :, :].reshape(-1, c)
        x_v_flat = x_v.flatten()
        y_v_flat = y_v.flatten()
        results['vertical'] = np.round(compute_correlation(x_v_flat, y_v_flat),4)

        # 对角方向
        x_d = img_float[0:m - 1, 0:n - 1, :].reshape(-1, c)
        y_d = img_float[1:m, 1:n, :].reshape(-1, c)
        x_d_flat = x_d.flatten()
        y_d_flat = y_d.flatten()
        results['diagonal'] = np.round(compute_correlation(x_d_flat, y_d_flat),4)

        # 通道间相关性 (R_G, R_B, G_B)
        R = img_float[:, :, 0].flatten()
        G = img_float[:, :, 1].flatten()
        B = img_float[:, :, 2].flatten()

        results['R_G'] = np.round(compute_correlation(R, G),4)
        results['R_B'] = np.round(compute_correlation(R, B),4)
        results['G_B'] = np.round(compute_correlation(G, B),4)

    else:
        # 灰度图像：只计算三个方向的相邻像素相关性
        m, n = img_float.shape

        # 水平方向
        x_h = img_float[:, 0:n - 1].flatten()
        y_h = img_float[:, 1:n].flatten()
        results['horizontal'] = np.round(compute_correlation(x_h, y_h),4)

        # 垂直方向
        x_v = img_float[0:m - 1, :].flatten()
        y_v = img_float[1:m, :].flatten()
        results['vertical'] = np.round(compute_correlation(x_v, y_v),4)

        # 对角方向
        x_d = img_float[0:m - 1, 0:n - 1].flatten()
        y_d = img_float[1:m, 1:n].flatten()
        results['diagonal'] = np.round(compute_correlation(x_d, y_d),4)

    return results


def resize(im,size):
    if len(im.shape)==2:
        img = cv.resize(im,size)
    else:
        img_r = np.expand_dims(cv.resize(im[:,:,0],size), 2)
        img_g = np.expand_dims(cv.resize(im[:, :, 1], size),2)
        img_b = np.expand_dims(cv.resize(im[:, :, 2], size),2)
        img = np.concatenate((img_r,img_g,img_b),2)

    return img


def csf():
    """
    Compute Contrast Sensitivity Function (CSF) matrix.
    """
    # Compute frequency response matrix
    Fmat = csfmat()

    # Compute 2-D filter coefficient using frequency sampling
    fc = fsamp2(Fmat)
    return fc
def csfmat():
    """
    Compute CSF frequency response matrix.
    """
    min_f = -20
    max_f = 20
    step_f = 1
    u = np.arange(min_f, max_f + step_f, step_f)
    v = np.arange(min_f, max_f + step_f, step_f)
    U, V = np.meshgrid(u, v)

    Z = np.zeros(U.shape)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            Z[i, j] = csffun(U[i, j], V[i, j])

    return Z
def csffun(u, v):
    """
    Contrast Sensitivity Function in spatial frequency.
    """
    sigma = 2
    f = np.sqrt(u ** 2 + v ** 2)
    w = 2 * np.pi * f / 60
    Sw = 1.5 * np.exp(-sigma ** 2 * w ** 2 / 2) - np.exp(-2 * sigma ** 2 * w ** 2 / 2)

    # Modification in high frequency
    sita = np.arctan2(v, u)
    bita = 8
    f0 = 11.13
    w0 = 2 * np.pi * f0 / 60
    Ow = (1 + np.exp(bita * (w - w0)) * (np.cos(2 * sita)) ** 4) / (1 + np.exp(bita * (w - w0)))

    # Compute final response
    Sa = Sw * Ow
    return Sa
def fsamp2(hd, f1=None, f2=None, siz=None):
    """
    Design a 2-D FIR filter using frequency sampling.

    Parameters
    ----------
    f1 : numpy.ndarray or list
        Vector or matrix of frequency samples in the x direction.
    f2 : numpy.ndarray or list
        Vector or matrix of frequency samples in the y direction.
    hd : numpy.ndarray
        Desired frequency response sampled at the points specified by f1 and f2.
    siz : tuple, optional
        Size of the FIR filter to be created (M, N). If None, the filter size is taken from HD.

    Returns
    -------
    numpy.ndarray
        The 2-D FIR filter coefficients.
    """
    if siz is None:
        # Uniform spacing case (fast)
        hd = np.rot90(fftshift(np.rot90(hd, 2)), 2)
        h = fftshift(ifft2(hd))

    else:
        # Create filter of size siz to solve problem at the points (f1, f2, hd)
        hd = np.array(hd, dtype=np.double)

        # Expand f1 and f2 if they are vectors.
        if np.issubdtype(type(f1), np.number) and np.issubdtype(type(f2), np.number):
            f1, f2 = np.meshgrid(f1, f2)

        if hd.size < np.prod(siz):
            raise Warning('Not enough frequency points.')

        # Convert frequency to radians.
        f1 = f1 * np.pi
        f2 = f2 * np.pi

        h = np.zeros(siz, dtype=np.complex)
        n1, n2 = np.meshgrid(np.arange(siz[1]) - siz[1] // 2, np.arange(siz[0]) - siz[0] // 2)
        DFT = np.exp(-1j * f1.ravel()[:, np.newaxis] * n1.ravel()[np.newaxis, :]) * np.exp(
            -1j * f2.ravel()[:, np.newaxis] * n2.ravel()[np.newaxis, :])
        h, _ = lstsq(DFT, hd.ravel())
        h = h.ravel()

        # Convert to real if possible.
    if np.all(np.abs(np.imag(h)) < np.sqrt(np.finfo(float).eps)):
        h = np.real(h)

    h = np.rot90(h, 2)  # Rotate for use with filter2

    return h
def wpsnr(A, B, fc=None):
    """
    Compute Weighted Peak Signal-to-Noise Ratio (WPSNR) between two images.
    """
    if np.array_equal(A, B):
        raise ValueError('Images are identical: PSNR has infinite value')

    max2_A = np.max(A)
    max2_B = np.max(B)
    min2_A = np.min(A)
    min2_B = np.min(B)

    if max2_A > 1 or max2_B > 1 or min2_A < 0 or min2_B < 0:
        raise ValueError('Input matrices must have values in the interval [0, 1]')

    e = A - B
    if fc is None:
        fc = csf()

    ew = signal.convolve2d(e, fc, mode='same', boundary='wrap')

    decibels = 20 * np.log10(1 / (np.sqrt(np.mean(ew ** 2))))
    return decibels