"""
压缩，直方图重组，每位bit置乱，自适应嵌入（8组，加上组内分解直方图重组），修正
"""

import math
import cv2 as cv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import hashlib
import time
import torch
import itertools
from pre_utils4 import IntegerLiftingW
from model2.cs_1d2_dct_fil2 import model as model_gray
from model2.cs_1d2_dct_fil2_color import model as model_color


def dec2bin(a):
    # if a.ndim == 3:
    #     [pages, rows, columns] = a.shape
    # else:
    #     [rows, columns] = a.shape
    #     pages = 1
    #
    # a = a.reshape(pages * rows * columns, 1)
    a = a.reshape(-1,1)

    b = a // 128
    a = a - b * 128
    b1 = a // 64
    a = a - b1 * 64
    b2 = a // 32
    a = a - b2 * 32
    b3 = a // 16
    a = a - b3 * 16
    b4 = a // 8
    a = a - b4 * 8
    b5 = a // 4
    a = a - b5 * 4
    b6 = a // 2
    a = a - b6 * 2
    # c = np.concatenate((b, b1, b2, b3, b4, b5, b6, a), axis=1)
    c = np.concatenate((a, b6, b5, b4, b3, b2, b1, b), axis=1)
    return c

def bin2dec(tem13):
    # tem11 = np.array([[128], [64], [32], [16], [8], [4], [2], [1]])
    tem11 = np.array([[1], [2], [4], [8], [16], [32], [64], [128]])
    tem12 = np.dot(tem13, tem11)
    return tem12

def Histogram_recombination(image,way):
    tem = np.copy(image).reshape(-1)
    aa = np.zeros(256)
    for x_value in range(256):
        aa[x_value] = float(tem[tem == x_value].shape[0])
    # print(aa)
    index_aa = np.argsort(-aa)

    # 按照前waybit0多原则
    if way==3: # 写死减少加密时间
       index_a = [0, 128, 64, 32, 16, 8, 192, 160, 144, 136, 96, 80, 72, 48, 40, 24, 224, 208, 200, 176, 168, 152, 112,
                   104, 88, 56, 240, 232, 216, 184, 120, 248, 4, 2, 1, 132, 130, 129, 68, 66, 65, 36, 34, 33, 20, 18, 17,
                   12, 10, 9, 196, 194, 193, 164, 162, 161, 148, 146, 145, 140, 138, 137, 100, 98, 97, 84, 82, 81, 76, 74,
                   73, 52, 50, 49, 44, 42, 41, 28, 26, 25, 228, 226, 225, 212, 210, 209, 204, 202, 201, 180, 178, 177, 172,
                   170, 169, 156, 154, 153, 116, 114, 113, 108, 106, 105, 92, 90, 89, 60, 58, 57, 244, 242, 241, 236, 234,
                   233, 220, 218, 217, 188, 186, 185, 124, 122, 121, 252, 250, 249, 6, 5, 3, 134, 133, 131, 70, 69, 67, 38,
                   37, 35, 22, 21, 19, 14, 13, 11, 198, 197, 195, 166, 165, 163, 150, 149, 147, 142, 141, 139, 102, 101, 99,
                   86, 85, 83, 78, 77, 75, 54, 53, 51, 46, 45, 43, 30, 29, 27, 230, 229, 227, 214, 213, 211, 206, 205, 203,
                   182, 181, 179, 174, 173, 171, 158, 157, 155, 118, 117, 115, 110, 109, 107, 94, 93, 91, 62, 61, 59, 246,
                   245, 243, 238, 237, 235, 222, 221, 219, 190, 189, 187, 126, 125, 123, 254, 253, 251, 7, 135, 71, 39, 23,
                   15, 199, 167, 151, 143, 103, 87, 79, 55, 47, 31, 231, 215, 207, 183, 175, 159, 119, 111, 95, 63, 247,
                   239, 223, 191, 127, 255]
    else:
        a = np.uint8(np.linspace(0, 255, 256))
        bin_a = dec2bin(a)
        bin_b1 = (1+way) - np.sum(bin_a[:,0:way],1)
        bin_b2 = (9-way) - np.sum(bin_a[:,way:],1)
        index_a = np.lexsort((-a, -bin_b2, -bin_b1))


    # df = pd.DataFrame({
    #     'matrix1': bin_b1,
    #     'matrix2': bin_b2,
    #     'original_index': range(len(bin_b1))
    # })
    # # 排序并获取索引
    # df_sorted = df.sort_values(['matrix1', 'matrix2', 'original_index'])
    # sorted_indices = df_sorted['original_index'].values

    tem2 = np.copy(image)
    for i in range(256):
        tem2[image == index_aa[i]] = index_a[i]

    # tem2_bin = dec2bin(tem2)
    # for i in range(8):
    #     print(np.sum(tem2_bin[:,i]))

    return tem2, index_aa

def Reverse_Histogram_recombination(image, way, index_aa):

    tem = np.copy(image)

    if way==3: # 写死减少加密时间
       index_a = [0, 128, 64, 32, 16, 8, 192, 160, 144, 136, 96, 80, 72, 48, 40, 24, 224, 208, 200, 176, 168, 152, 112,
                   104, 88, 56, 240, 232, 216, 184, 120, 248, 4, 2, 1, 132, 130, 129, 68, 66, 65, 36, 34, 33, 20, 18, 17,
                   12, 10, 9, 196, 194, 193, 164, 162, 161, 148, 146, 145, 140, 138, 137, 100, 98, 97, 84, 82, 81, 76, 74,
                   73, 52, 50, 49, 44, 42, 41, 28, 26, 25, 228, 226, 225, 212, 210, 209, 204, 202, 201, 180, 178, 177, 172,
                   170, 169, 156, 154, 153, 116, 114, 113, 108, 106, 105, 92, 90, 89, 60, 58, 57, 244, 242, 241, 236, 234,
                   233, 220, 218, 217, 188, 186, 185, 124, 122, 121, 252, 250, 249, 6, 5, 3, 134, 133, 131, 70, 69, 67, 38,
                   37, 35, 22, 21, 19, 14, 13, 11, 198, 197, 195, 166, 165, 163, 150, 149, 147, 142, 141, 139, 102, 101, 99,
                   86, 85, 83, 78, 77, 75, 54, 53, 51, 46, 45, 43, 30, 29, 27, 230, 229, 227, 214, 213, 211, 206, 205, 203,
                   182, 181, 179, 174, 173, 171, 158, 157, 155, 118, 117, 115, 110, 109, 107, 94, 93, 91, 62, 61, 59, 246,
                   245, 243, 238, 237, 235, 222, 221, 219, 190, 189, 187, 126, 125, 123, 254, 253, 251, 7, 135, 71, 39, 23,
                   15, 199, 167, 151, 143, 103, 87, 79, 55, 47, 31, 231, 215, 207, 183, 175, 159, 119, 111, 95, 63, 247,
                   239, 223, 191, 127, 255]
    else:
        a = np.uint8(np.linspace(0, 255, 256))
        bin_a = dec2bin(a)
        bin_b1 = (1+way) - np.sum(bin_a[:,0:way],1)
        bin_b2 = (9-way) - np.sum(bin_a[:,way:],1)
        index_a = np.lexsort((-a, -bin_b2, -bin_b1))

    for i in range(256):
        tem[image == index_a[i]] = index_aa[i]

    return tem

def cor(cip,cover,e,spa=False):
    tem = cip-cover
    tem1 = np.copy(cip)
    tem1[tem > e/2] = cip[tem>e/2] - e
    tem1[tem <- e/2] = cip[tem <- e/2] + e
    if spa==True:
        tem1[tem1 > 255] = cip[tem1 > 255]
        tem1[tem1 < 0] = cip[tem1 < 0]
    return tem1


def vm(image, a):

    if a>2:
        b = np.uint8(np.linspace(0, a-1, a))
        bin_b = dec2bin(b)
        non_zero_cols = np.any(bin_b != 0, axis=0)
        first_non_zero = np.argmax(non_zero_cols)
        bin_b = bin_b[:, first_non_zero:].tolist()
        inder_sort_b = np.argsort(np.sum(bin_b,1))

        ideal = np.copy(b)
        ideal[b>a/2] = a-ideal[b>a/2]
        ideal = np.argsort(ideal)

        tem2 = np.copy(image)
        for i in range(a):
            tem2[image == inder_sort_b[i]] = ideal[i]

    else:
        tem2 = np.copy(image)

    return tem2

def dvm(tem2, a):
    if a > 2:
        b = np.uint8(np.linspace(0, a-1, a))
        bin_b = dec2bin(b)
        non_zero_cols = np.any(bin_b != 0, axis=0)
        first_non_zero = np.argmax(non_zero_cols)
        bin_b = bin_b[:, first_non_zero:].tolist()
        inder_sort_b = np.argsort(np.sum(bin_b, 1))

        ideal = np.copy(b)
        ideal[b>a/2] = a-ideal[b>a/2]
        ideal = np.argsort(ideal)

        rim = np.copy(tem2)
        for i in range(a):
            rim[tem2 == ideal[i]] = inder_sort_b[i]
            # rim[tem2 == inder_sort_b[i]] = ideal[i]
    else:
        rim = np.copy(tem2)

    return rim

def ss(tem,miu):
    "底层函数"
    tem = np.mod((miu*math.e**math.pi)/np.sin((math.e*7)/(tem+10**-14)),1)
    return tem

def IICM_chaos(inti, miu, T):
    """
    :param Initial_value: 混沌初值
    :param parameter: 混沌参数
    :param N: 空间维度
    :param T: 迭代代数
    :return: N*T的混沌序列
    其中时空混沌的第一个网格由logistc混沌生产
    """
    tem_int = np.zeros(210)
    tem_int[0] = inti
    for i in range(209):
        tem_int[i + 1] = np.sqrt(1 - tem_int[i] ** 2) * np.sin(miu[0] / (tem_int[i] ** 2))

    jj = np.argsort(tem_int[10:110])
    kk = np.argsort(tem_int[110:210])
    num_clc = int(np.ceil((T * 3) / 100))
    tem = np.zeros([100, num_clc+100])
    tem[:, 0] = (tem_int[10:110] + tem_int[110:210]) / 2

    miu = miu[1:101]
    for i in range(num_clc + 99):
        tem[:, i + 1] = np.sqrt(1 - tem[:, i] ** 2) * np.sin(miu / (tem[jj, i] * tem[kk, i]))

    tem = tem[:, 100:].flatten()
    tem = tem[0: int(3*int(num_clc/3 * 100))].reshape(3, int(num_clc/3 * 100))

    return tem

def im_sha256(im, sec_key):
    # 当前时间与输入图像的哈希值
    # 动态密钥产生

    # 保证数组在内存中连续，避免 hashlib.sha256 报错
    im = np.ascontiguousarray(im)

    # 图像sha值提取
    if len(im.shape) == 3:
        H1 = hashlib.sha256(im).hexdigest()
    else:
        H1 = hashlib.sha256(im).hexdigest()

    # 当前时间sha值提取
    T = time.time()
    H2 = hashlib.sha256(np.array(T)).hexdigest()

    # 安全密钥H1、H2xor处理得到H
    H = np.zeros([32, ])
    for i in range(32):
        H[i] = eval('0x' + H1[2 * i:2 * (i + 1)]) ^ eval('0x' + H2[2 * i:2 * (i + 1)]) ^ eval('0x' + sec_key[2 * i:2 * (i + 1)])

    H = np.array(H, np.uint8)
    h = np.sum(H) / 8192

    numbers = list(range(0, 31))  # 1到32
    combinations = list(itertools.combinations(numbers, 2))

    inti = h
    miu = np.zeros(101)
    for i in range(101):
        miu[i] = 50 * h * (H[combinations[i][0]] ^ H[combinations[i][1]])/255 + 5

    dynamic_key = [inti,miu]

    return dynamic_key

def bit_3Dscramble(im, rr1, rr2):
    """
    :param im: 输入图像
    :param r: 伪随机数
    :return: 置乱后的图像
    每位得bit分开置乱
    """
    if len(im.shape) == 2:
        [M, N] = im.shape
        K = 1
        k, m, n = 16, int(M / 4), int(N / 4)
    else:
        [M, N, K] = im.shape
        k, m, n = 16 * K, int(M / 4), int(N / 4)

    bit_im = dec2bin(im)

    # bit级别的置乱 堆叠三次
    for j in range(8):
        tem = bit_im[:, j].reshape([k, m, n])
        r1 = np.argsort(rr1[j * (k + m + n):(j + 1) * (k + m + n)]).astype(int)
        r2 = np.mod(rr2[128 * j * (k + m + n):128 * (j + 1) * (k + m + n)] * 10 ** 10, 64).astype(int)
        for i in range(len(r1)):
            if r1[i] < k:
                r = r2[128 * i:128 * i + m + n]
                t = r1[i]
                tem[t, :, :] = np.roll(tem[t, :, :], r[:m], 0)
                tem[t, :, :] = np.roll(tem[t, :, :], r[m:], 1)
            elif r1[i] < k + m:
                r = r2[128 * i: 128 * i + k + n]
                t = r1[i] - k
                tem[:, t, :] = np.roll(tem[:, t, :], r[:k], 0)
                tem[:, t, :] = np.roll(tem[:, t, :], r[k:], 1)
            else:
                r = r2[128 * i: 128 * i + k + m]
                t = r1[i] - k - m
                tem[:, :, t] = np.roll(tem[:, :, t], r[:k], 0)
                tem[:, :, t] = np.roll(tem[:, :, t], r[k:], 1)
        bit_im[:, j] = tem.reshape(-1)
    if K == 1:
        cip = bin2dec(bit_im).reshape([M, N])
    else:
        cip = bin2dec(bit_im).reshape([M, N, K])

    return bit_im, cip
def bit_3Ddescramble(cip, rr1, rr2):
    """
    :param cip: 输入图像
    :param r: 伪随机数
    :return: 置乱解密的图像
    :cp: 0,1互换的位置记录
    将0bit比较重的前4位一起组成一个3D矩阵，后面分布比较均匀的组成一个3D矩阵分别置乱
    先将每位的bit按照0，1比重转化多的位0
    """

    if len(cip.shape) == 2:
        [M, N] = cip.shape
        K = 1
        k, m, n = 16, int(M / 4), int(N / 4)
    else:
        [M, N, K] = cip.shape
        k, m, n = 16 * K, int(M / 4), int(N / 4)

    bit_cip = dec2bin(cip)
    for j in range(8):
        tem = bit_cip[:, j].reshape([k, m, n])
        r1 = np.argsort(rr1[j * (k + m + n):(j + 1) * (k + m + n)]).astype(int)
        r2 = np.mod(rr2[128 * j * (k + m + n):128 * (j + 1) * (k + m + n)] * 10 ** 10, 64).astype(int)
        for i in range(len(r1) - 1, -1, -1):
            if r1[i] < k:
                r = -r2[128 * i:128 * i + m + n]
                t = r1[i]
                tem[t, :, :] = np.roll(tem[t, :, :], r[:m], 0)
                tem[t, :, :] = np.roll(tem[t, :, :], r[m:], 1)
            elif r1[i] < k + m:
                r = -r2[128 * i: 128 * i + k + n]
                t = r1[i] - k
                tem[:, t, :] = np.roll(tem[:, t, :], r[:k], 0)
                tem[:, t, :] = np.roll(tem[:, t, :], r[k:], 1)
            else:
                r = -r2[128 * i: 128 * i + k + m]
                t = r1[i] - k - m
                tem[:, :, t] = np.roll(tem[:, :, t], r[:k], 0)
                tem[:, :, t] = np.roll(tem[:, :, t], r[k:], 1)
        bit_cip[:, j] = tem.reshape(-1)

    if K == 1:
        rim = bin2dec(bit_cip).reshape([M, N])
    else:
        rim = bin2dec(bit_cip).reshape([M, N, K])

    return rim

def edge_sort(im,sort_edges,ca2,a,spa=False):
    "对边缘排序后将值较大的嵌入到边缘区域，值较小的嵌入到非边缘区域"
    # ca2 为 1*mnk
    # time1 = time.time()
    if a==0:
        ca4 = im
    else:
        [m,n]=ca2.shape[:2]

        ca1 = im.ravel()[sort_edges] #按照sort_edges对im像素重新排序
        ca3 = np.copy(ca1)
        ca3[:m*n] = cor(np.mod(ca2.ravel() + np.mod(ca1[:m*n], a), a) + np.floor(ca1[:m*n] / a) * a, ca1[:m*n], a, spa)
        ca4 = np.zeros(im.shape).ravel()
        ca4[sort_edges] = ca3
        ca4 = ca4.reshape(im.shape)

    return ca4
def de_edge_sort(cover,sort_edges,cip,a,im_size):
    "对边缘排序后将值较大的嵌入到边缘区域，值较小的嵌入到非边缘区域"

    if a==0:
        ca2 = np.zeros_like(cip)
    else:
        if len(im_size) == 2:
            [m, n] = im_size
            k = 1
        else:
            [m, n, k] = im_size

        ca1 = cover.ravel()[sort_edges]
        ca3 = cip.ravel()[sort_edges]

        ca2 = np.mod(np.mod(ca3[:m*n*k],a)-np.mod(ca1[:m*n*k],a),a)
        ca2 = ca2.reshape(im_size)

    return ca2
def Laplacian_embed(nc,cover, kt='haar', dco=[2,4,4,8]):
    "通过Laplacian边缘检测，将cover每个频段信息划分为边缘与非边缘区域"
    cover[cover > 250] = 250
    cover[cover < 5] = 5
    # [m,n] = nc.shape[:2]
    m, n, k = cover.shape if cover.ndim == 3 else (*cover.shape, 1)
    [a, b, c, d] = dco

    # 整数提升小波变换
    ls = IntegerLiftingW.from_name(kt)
    cover = cover.astype(np.int32)
    if k == 1:
        ca1, ch1, cv1, cd1 = ls.lwt2(cover)
    else:
        CAs, CHs, CVs, CDs = [], [], [], []
        for i in range(k):
            cac, chc, cvc, cdc = ls.lwt2(cover[:, :, i])
            CAs.append(cac)
            CHs.append(chc)
            CVs.append(cvc)
            CDs.append(cdc)
        ca1 = np.stack(CAs, axis=2)
        ch1 = np.stack(CHs, axis=2)
        cv1 = np.stack(CVs, axis=2)
        cd1 = np.stack(CDs, axis=2)

    # 整数小波变换分解
    # tem = IWT_haar(cover)
    # ca1 = tem[:int(M / 2), :int(N / 2)]
    # ch1 = tem[int(M / 2):, :int(N / 2)]
    # cv1 = tem[:int(M / 2), int(N / 2):]
    # cd1 = tem[int(M / 2):, int(N / 2):]

    # cv.imshow('ca', np.uint8(abs(ca1)))
    # cv.imshow('ch', np.uint8(abs(ch1)))
    # cv.imshow('cv', np.uint8(abs(cv1)))
    # cv.imshow('cd', np.uint8(abs(cd1)))

    ## 求取边缘像素点
    ca1_gau = cv.GaussianBlur(np.uint8(cover[0:m:2, 0:n:2]), (3, 3), sigmaX=0)
    im_edges = cv.Laplacian(ca1_gau, cv.CV_16S, ksize=3)


    # im_edges = abs(cd1)
    # cv.imshow('im_edges',np.uint8(abs(im_edges)))
    sort_edges = np.argsort(-(abs(im_edges)).ravel())   #按照最有可能为边缘到最没有可能为封面的顺序排列


    # nc重新整理bit的0，1分布
    bit_nc = dec2bin(nc)
    # print(np.sum(bit_nc,0))


    # 1由多到少
    bit_nc07 = np.flip(bit_nc)
    bit_nc07 = np.reshape(bit_nc07.T, [-1, 8])  # 前面的后几位有更多的1
    nc = bin2dec(bit_nc07)

    # 拆分NC
    ca2 = np.floor(nc / (b * c * d))
    ch2 = np.floor(np.mod(nc, b * c * d) / (c * d))
    cv2 = np.floor(np.mod(nc, c * d) / d)
    cd2 = np.mod(nc, d)

    # print(np.sum(ca2), np.sum(ch2), np.sum(cv2), np.sum(cd2))

    # # 修改拆分后的不同频段（这个应该根据0，1bit重组来进行）
    ca2 = vm(ca2, a)
    ch2 = vm(ch2, b)
    cv2 = vm(cv2, c)
    cd2 = vm(cd2, d)


    # print(np.sum(ca2),np.sum(ch2),np.sum(cv2),np.sum(cd2))


    ca3 = edge_sort(ca1, sort_edges, ca2, a)
    ch3 = edge_sort(ch1, sort_edges, ch2, b)
    cv3 = edge_sort(cv1, sort_edges, cv2, c)
    cd3 = edge_sort(cd1, sort_edges, cd2, d)

    # # 逆整数小波
    # tem1 = np.concatenate((ca3, ch3), 0)
    # tem2 = np.concatenate((cv3, cd3), 0)
    # tem3 = np.concatenate((tem1, tem2), 1)
    # cip = DIWT_haar(tem3)

    # 合并
    if k == 1:
        cip = ls.ilwt2(ca3, ch3, cv3, cd3)
    else:
        cips = []
        for i in range(k):
            cipc = ls.ilwt2(ca3[:, :, i], ch3[:, :, i], cv3[:, :, i], cd3[:, :, i])
            cips.append(cipc)
        cip = np.stack(cips, axis=2)

    return np.clip(cip,0,255)

def Laplacian_extract(cip, cover, im_size, kt='haar', dco=[2,4,4,8]):
    "通过Laplacian边缘检测，将cover每个频段信息划分为边缘与非边缘区域"
    cover[cover > 250] = 250
    cover[cover < 5] = 5
    m, n, k = cover.shape if cover.ndim == 3 else (*cover.shape, 1)
    # [M,N] = cover.shape[:2]
    # m, n = im_size
    [a, b, c, d] = dco
    # ## 整数小波变换分解
    # # 封面
    # tem = IWT_haar(cover)
    # ca1 = tem[:int(M / 2), :int(N / 2)]
    # ch1 = tem[int(M / 2):, :int(N / 2)]
    # cv1 = tem[:int(M / 2), int(N / 2):]
    # cd1 = tem[int(M / 2):, int(N / 2):]
    # # 密文
    # tem = IWT_haar(cip)
    # ca3 = tem[:int(M / 2), :int(N / 2)]
    # ch3 = tem[int(M / 2):, :int(N / 2)]
    # cv3 = tem[:int(M / 2), int(N / 2):]
    # cd3 = tem[int(M / 2):, int(N / 2):]


    # 整数提升小波变换
    ls = IntegerLiftingW.from_name(kt)
    cover = cover.astype(np.int32)
    cip = cip.astype(np.int32)
    if k == 1:
        ca1, ch1, cv1, cd1 = ls.lwt2(cover)
        ca3, ch3, cv3, cd3 = ls.lwt2(cip)
    else:
        CAs, CHs, CVs, CDs = [], [], [], []
        for i in range(k):
            cac, chc, cvc, cdc = ls.lwt2(cover[:, :, i])
            CAs.append(cac)
            CHs.append(chc)
            CVs.append(cvc)
            CDs.append(cdc)
        ca1 = np.stack(CAs, axis=2)
        ch1 = np.stack(CHs, axis=2)
        cv1 = np.stack(CVs, axis=2)
        cd1 = np.stack(CDs, axis=2)

        CAs, CHs, CVs, CDs = [], [], [], []
        for i in range(k):
            cac, chc, cvc, cdc = ls.lwt2(cip[:, :, i])
            CAs.append(cac)
            CHs.append(chc)
            CVs.append(cvc)
            CDs.append(cdc)
        ca3 = np.stack(CAs, axis=2)
        ch3 = np.stack(CHs, axis=2)
        cv3 = np.stack(CVs, axis=2)
        cd3 = np.stack(CDs, axis=2)

    ## 求取边缘像素点
    # ca进行边缘检测
    # ca_edges = abs(cv.Laplacian(np.uint8(ca1), cv.CV_16S, ksize=1))
    # 对整个封面进行边缘检测，再缩放到ca大小，再次二值化（保证更多点为边缘）

    img = cv.GaussianBlur(np.uint8(cover[0:m:2, 0:n:2]), (3, 3), sigmaX=0)
    im_edges = cv.Laplacian(img, cv.CV_16S, ksize=3)

    # im_edges = abs(cd1)
    # cv.imshow('im_edges',np.uint8(abs(im_edges)))
    # ca_edges = abs(cv.resize(im_edges,ca1.shape))
    sort_edges = np.argsort(-(abs(im_edges)).ravel())   #按照最有可能为边缘到最没有可能为封面的顺序排列


    # 提取
    # ca2 = np.mod(np.mod(ca3,a)-np.mod(ca1,a),a).ravel()
    ca2 = de_edge_sort(ca1, sort_edges, ca3, a, im_size)
    ch2 = de_edge_sort(ch1, sort_edges, ch3, b, im_size)
    cv2 = de_edge_sort(cv1, sort_edges, cv3, c, im_size)
    cd2 = de_edge_sort(cd1, sort_edges, cd3, d, im_size)

    # 反向修正
    ca2 = dvm(ca2, a)
    ch2 = dvm(ch2, b)
    cv2 = dvm(cv2, c)
    cd2 = dvm(cd2, d)

    # 对重新整理的bit归位
    nc = ca2*(b * c * d) + ch2*(c * d) + cv2*(d) + cd2
    bit_nc = dec2bin(nc)

    # 1由多到少
    bit_nc = np.reshape(bit_nc,[8,-1]).T
    bit_nc = np.flip(bit_nc)
    nc = bin2dec(bit_nc).reshape(im_size)

    return nc
def Laplacian_embed_spa(nc, cover, dco=[4,4,4,4]):

    # [m,n] = nc.shape[:2]
    [M,N] = cover.shape[:2]
    [a, b, c, d] = dco
    # 整数小波变换分解
    tem = cover
    ca1 = tem[0:M:2, 0:N:2]
    ch1 = tem[1:M:2, 0:N:2]
    cv1 = tem[0:M:2, 1:N:2]
    cd1 = tem[1:M:2, 1:N:2]
    ## 求取边缘像素点
    # ca进行边缘检测
    # ca_edges = abs(cv.Laplacian(np.uint8(ca1), cv.CV_16S, ksize=1))
    # 对整个封面进行边缘检测，再缩放到ca大小，再次二值化（保证更多点为边缘）

    img = cv.GaussianBlur(np.uint8(ca1), (3, 3), sigmaX=0)
    im_edges = cv.Laplacian(img, cv.CV_16S, ksize=3)


    # cv.imshow('im_edges',np.uint8(abs(im_edges)))
    # ca_edges = abs(cv.resize(im_edges,ca1.shape))
    sort_edges = np.argsort(-(abs(im_edges)).ravel())   #按照最有可能为边缘到最没有可能为封面的顺序排列


    # nc重新整理bit的0，1分布
    bit_nc = dec2bin(nc)
    # print(np.sum(bit_nc,0))


    # 1由多到少
    bit_nc07 = np.flip(bit_nc)
    bit_nc07 = np.reshape(bit_nc07.T, [-1, 8])  # 前面的后几位有更多的1
    nc = bin2dec(bit_nc07)

    # 拆分NC
    ca2 = np.floor(nc / (b * c * d))
    ch2 = np.floor(np.mod(nc, b * c * d) / (c * d))
    cv2 = np.floor(np.mod(nc, c * d) / d)
    cd2 = np.mod(nc, d)

    # print(np.sum(ca2), np.sum(ch2), np.sum(cv2), np.sum(cd2))

    ca2 = vm(ca2, a)
    ch2 = vm(ch2, b)
    cv2 = vm(cv2, c)
    cd2 = vm(cd2, d)

    # print(np.sum(ca2),np.sum(ch2),np.sum(cv2),np.sum(cd2))


    ca3 = edge_sort(ca1, sort_edges, ca2, a, True)
    ch3 = edge_sort(ch1, sort_edges, ch2, b, True)
    cv3 = edge_sort(cv1, sort_edges, cv2, c, True)
    cd3 = edge_sort(cd1, sort_edges, cd2, d, True)

    # 逆整数小波
    cip = np.ones_like(cover)
    cip[0:M:2, 0:N:2] = ca3
    cip[1:M:2, 0:N:2] = ch3
    cip[0:M:2, 1:N:2] = cv3
    cip[1:M:2, 1:N:2] = cd3

    return np.clip(cip,0,255)
def Laplacian_extract_spa(cip, cover, im_size, dco=[4,4,4,4]):
    "通过Laplacian边缘检测，将cover每个频段信息划分为边缘与非边缘区域"
    [M,N] = cover.shape[:2]
    # m, n = im_size
    [a, b, c, d] = dco
    ## 整数小波变换分解
    # 封面
    tem = cover
    ca1 = tem[0:M:2, 0:N:2]
    ch1 = tem[1:M:2, 0:N:2]
    cv1 = tem[0:M:2, 1:N:2]
    cd1 = tem[1:M:2, 1:N:2]
    # 密文
    tem = cip
    ca3 = tem[0:M:2, 0:N:2]
    ch3 = tem[1:M:2, 0:N:2]
    cv3 = tem[0:M:2, 1:N:2]
    cd3 = tem[1:M:2, 1:N:2]

    ## 求取边缘像素点
    # ca进行边缘检测
    # ca_edges = abs(cv.Laplacian(np.uint8(ca1), cv.CV_16S, ksize=1))
    # 对整个封面进行边缘检测，再缩放到ca大小，再次二值化（保证更多点为边缘）

    img = cv.GaussianBlur(np.uint8(ca1), (3, 3), sigmaX=0)
    im_edges = cv.Laplacian(img, cv.CV_16S, ksize=3)

    # cv.imshow('im_edges',np.uint8(abs(im_edges)))
    # ca_edges = abs(cv.resize(im_edges,ca1.shape))
    sort_edges = np.argsort(-(abs(im_edges)).ravel())   #按照最有可能为边缘到最没有可能为封面的顺序排列


    # 提取
    # ca2 = np.mod(np.mod(ca3,a)-np.mod(ca1,a),a).ravel()
    ca2 = de_edge_sort(ca1, sort_edges, ca3, a, im_size)
    ch2 = de_edge_sort(ch1, sort_edges, ch3, b, im_size)
    cv2 = de_edge_sort(cv1, sort_edges, cv3, c, im_size)
    cd2 = de_edge_sort(cd1, sort_edges, cd3, d, im_size)

    # 反向修正
    ca2 = dvm(ca2, a)
    ch2 = dvm(ch2, b)
    cv2 = dvm(cv2, c)
    cd2 = dvm(cd2, d)

    # 对重新整理的bit归位
    nc = ca2*(b * c * d) + ch2*(c * d) + cv2*(d) + cd2
    bit_nc = dec2bin(nc)

    # 1由多到少
    bit_nc = np.reshape(bit_nc,[8,-1]).T
    bit_nc = np.flip(bit_nc)
    nc = bin2dec(bit_nc).reshape(im_size)

    return nc

def load_model(path, in_channels=1, device='cpu'):
    """
    加载 model2 目录下的灰度或彩色模型，并自动处理 state_dict 前缀。
    """
    if in_channels == 1:
        net = model_gray(1, embed_dim=[64, 128, 256, 256, 128, 64], depth=[5, 5, 5, 5, 5, 5])
    elif in_channels == 3:
        net = model_color(3, embed_dim=[64, 128, 256, 256, 128, 64], depth=[5, 5, 5, 5, 5, 5])
    else:
        raise ValueError(f"不支持的通道数: {in_channels}")

    checkpoint = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('_orig_mod.module.'):
            new_key = k[len('_orig_mod.module.'):]
        elif k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    del checkpoint
    del new_state_dict
    return net

def encryption(im, cover, sec_key, hrp, emb_type, model, dco=[4,4,4,4]):
    """
    :param im:
    :param cover:
    :param sec_key:
    :param com_p: 压缩方式选择 1，压缩感知  2，分块小波
    :param hrp: bit排序方式选择 1~8
    :return:
    """
    if len(im.shape) == 2:
        [m, n] = im.shape
        k = 1
    else:
        [m, n, k] = im.shape

    t1 = time.time()

    dynamic_key = im_sha256(im, sec_key)
    miu = dynamic_key[1]
    inti = dynamic_key[0]
    r = IICM_chaos(inti, miu, max([(k * 1024 * 1024),int(n*m*k)]))      #########

    # 深度分块压缩感知
    if k==1:
        r1 = torch.argsort(torch.tensor(r[0, :int(m * n)]).view(-1))
        r1 = r1.repeat(1, 1)
        U, S, V = torch.linalg.svd(torch.tensor(r[1, :int(1024 * 1024)]).view(1024, 1024))
        Phi = (U @ V)[:, :256].float()

        img = torch.tensor(im / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            Y = model.comp(img, Phi, r1).squeeze()

    else:
        phi_list = []
        perm_list = []
        m_all = [462, 153, 153]
        for i in range(k):
            mi = m_all[i]
            # 每个通道 1024x1024 测量矩阵
            mat = torch.tensor(r[1, i * 1024 * 1024:(i + 1) * 1024 * 1024]).view(1024, 1024)
            U, S, V = torch.linalg.svd(mat)
            Phi = (U @ V)[:, :mi].float()
            phi_list.append(Phi.unsqueeze(0))

            # 每个通道独立置乱序列
            seq = torch.tensor(r[0, i * m * n:(i + 1) * m * n])
            perm_i = torch.argsort(seq).unsqueeze(0)
            perm_list.append(perm_i)

        img = torch.tensor(im / 255.0, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad():
            Y = model.comp(img, phi_list, perm_list).squeeze()

    max_x2 = torch.max(Y)
    min_x2 = torch.min(Y)
    x1 = np.array(torch.round((Y - min_x2) / (max_x2 - min_x2) * 255))
    max_min = [max_x2, min_x2]
    if k == 1:
        x1 = x1.reshape(int(m/2),int(n/2))
    else:
        x1 = x1.reshape(int(m / 2), int(n / 2), k)

    # bit_x1 = dec2bin(x1)
    # print('bit_x1',sum(sum(dec2bin(bit_x1))))
    # 直方图
    # print(sum(sum(dec2bin(x1))))
    # print(calc_ent(x1.reshape(256, -1)))
    # plt_hist(x1, 1)
    cv.imshow('x1', np.uint8(x1))

    x2, index_x1 = Histogram_recombination(x1,way=hrp)
    # plt_hist(x2, 2)
    # cv.imshow('x2', np.uint8(x2))
    # print(sum(sum(dec2bin(x2))))
    # bit_x2 = dec2bin(x2)
    # print('bit_x2', sum(sum(dec2bin(bit_x2))), sum(sum(dec2bin(bit_x2[:,:4]))), sum(sum(dec2bin(bit_x2[:,4:]))))

    # bit置乱（分8份）
    r3 = r[2, :int(n*m*k/2)]
    r4 = r[2, int(n*m*k/2):]
    bit_im, nc = bit_3Dscramble(x2,r3,r4)

    # bit_nc = dec2bin(nc)
    # print('bit_x2', sum(sum(dec2bin(bit_nc))), sum(sum(dec2bin(bit_nc[:, :4]))), sum(sum(dec2bin(bit_nc[:, 4:]))))
    # plt_hist(nc, 3)
    # cv.imshow('nc', np.uint8(nc))

    t2 = time.time()
    # 嵌入
    if emb_type == 'spa':
        cip = Laplacian_embed_spa(nc, cover, dco)
        # cip = Laplacian_embed(nc, cover, dco=[3, 5, 3, 6])
    else:
        cip = Laplacian_embed(nc, cover, emb_type, dco)

    t3 = time.time()
    # print(f"en_time: {t2 - t1:.4f}, emb_time: {t3 - t2:.4f}")
    return cip, nc, dynamic_key, max_min, index_x1

def dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, model, im_shape, dco=[4,4,4,4]):
    if len(im_shape) == 2:
        m, n = im_shape[0], im_shape[1]
        k = 1
        nc_size = [int(m / 2), int(n / 2)]
    else:
        m, n, k = im_shape[0], im_shape[1], im_shape[2]
        nc_size = [int(m / 2), int(n / 2), k]

    t1 = time.time()

    # 伪随机参数
    miu = dynamic_key[1]
    inti = dynamic_key[0]
    r = IICM_chaos(inti, miu, max([k * 1024 * 1024, int(n * m * k)]))

    # # 提取
    if emb_type == 'spa':
        nc = Laplacian_extract_spa(cip, cover, nc_size, dco)
        # nc = Laplacian_extract(cip, cover, nc_size, dco=[3, 5, 3, 6])
    else:
        nc = Laplacian_extract(cip, cover, nc_size, emb_type, dco)

    t2 = time.time()

    # 解置乱
    r3 = r[2, :int(n*m*k/2)]
    r4 = r[2, int(n*m*k/2):]
    x2 = bit_3Ddescramble(nc, r3, r4)

    # 直方图重组
    x1 = Reverse_Histogram_recombination(x2, way=hrp, index_aa=index_x1)

    # 重构
    if k == 1:
        x1 = x1.reshape(-1,256)
    else:
        x1 = x1.reshape(-1,256*3)

    Y = torch.tensor(x1).unsqueeze(0).float()
    max_x2, min_x2 = max_min[0], max_min[1]
    Y = Y / 255 * (max_x2 - min_x2) + min_x2

    if k == 1:
        r1 = torch.argsort(torch.tensor(r[0, :int(m * n)]).view(-1))
        r1_inv = torch.empty_like(r1)  # 生成一个大小与perm相同的矩阵
        r1_inv[r1] = torch.arange(r1.shape[0])  # 生成一个perm逆的矩阵。
        r1 = r1.repeat(1, 1)
        r1_inv = r1_inv.repeat(1, 1)
        U, S, V = torch.linalg.svd(torch.tensor(r[1, :int(1024 * 1024)]).view(1024, 1024))
        Phi = (U @ V)[:, :k * 256].float()
        Phit = torch.pinverse(Phi).float()

        with torch.no_grad():
            rim = model.refact(Y, Phi, Phit, r1, r1_inv, [1, k, m, n])
        rim = np.clip(np.array(rim.squeeze(0).squeeze(0)), 0, 1) * 255

    else:
        phi_list = []
        Phit_list = []
        perm_list = []
        perm_inv_list = []
        m_all = [462, 153, 153]
        for i in range(k):
            mi = m_all[i]
            # 每个通道 1024x1024 测量矩阵
            mat = torch.tensor(r[1, i * 1024 * 1024:(i + 1) * 1024 * 1024]).view(1024, 1024)
            U, S, V = torch.linalg.svd(mat)
            Phi = (U @ V)[:, :mi].float()
            Phit = torch.pinverse(Phi).float()
            phi_list.append(Phi.unsqueeze(0))
            Phit_list.append(Phit.unsqueeze(0))

            # 每个通道独立置乱序列
            seq = torch.tensor(r[0, i * m * n:(i + 1) * m * n])
            perm_i = torch.argsort(seq).unsqueeze(0)
            perm_inv_i = torch.argsort(perm_i, dim=-1)
            perm_list.append(perm_i)
            perm_inv_list.append(perm_inv_i)

        with torch.no_grad():
            rim = model.refact(Y, phi_list, Phit_list, perm_list, perm_inv_list, [1, k, m, n])
        rim = np.clip(np.array(rim.permute(0, 2, 3, 1).squeeze(0)), 0, 1) * 255

    t3 = time.time()
    # print(f"dec_time: {t2 - t1:.4f}, refact_time: {t3 - t2:.4f}")

    return rim, nc


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    import warnings
    warnings.filterwarnings("ignore")

    # im = cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/avion.ppm',0)
    # cover = cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/sailboat.ppm',0)
    # im = np.double(cv.resize(im,[512,512]))
    # cover = np.double(cv.resize(cover, [512, 512]))
    #
    # sec_key = '8d5ab8ba5340fce4420829ad5d12a0e45dacb0858544163d04c1d02b73e3697d'
    # hrp = 3  #
    # emb_types = ['spa', 'haar', 'db4', 'sym4', 'coif2', 'rbio3.7', 'cdf4.4', 'bs3']
    # dcos = [[4,4,4,4],[2,4,4,8],[8,4,4,2],[0,4,4,16],[0,0,2,128],[2,4,4,8],[0,4,4,16],[0,4,4,16]]
    # emb_type = emb_types[4]
    # dco = dcos[4]
    #
    # im_shape = im.shape
    # m, n, k = im.shape if im.ndim == 3 else (*im.shape, 1)
    # if k==1:
    #     module = load_model('model2/model_epoch_gray.pth', in_channels=1, device='cpu')
    # else:
    #     module = load_model('model2/model_epoch_color.pth', in_channels=3, device='cpu')
    #
    # cip, nc, dynamic_key, max_min, index_x1 = encryption(im, cover, sec_key, hrp, emb_type, module, dco)
    # # rim, nc2 = dencryption(cip, cover, dynamic_key, hrp, emb_type, max_min, index_x1, module, im_shape, dco)
    #
    # # print('nc-nc2:',np.sum(np.abs(nc - nc2)))
    #
    # print(PSNR(np.uint8(cip),np.uint8(cover)))
    # print(SSIM(np.uint8(cip), np.uint8(cover),multichannel=True))
    # # print(PSNR(np.uint8(im),np.uint8(rim)))
    # # print(SSIM(np.uint8(im),np.uint8(rim),multichannel=True))
    #
    # cv.imshow('im', np.uint8(im))
    # cv.imshow('cover', np.uint8(cover))
    # cv.imshow('nc', np.uint8(nc))
    # cv.imshow('cip', np.uint8(cip))
    # cv.imshow('cover-cip', np.uint8(abs(cover-cip)*50))
    # cv.imshow('cover-cip-2', np.uint8(255-abs(cover - cip) * 50))
    # # cv.imshow('rim', np.uint8(rim))
    # # cv.imshow('rim-im', np.uint8(abs(im-rim)*50))
    # plt.show()
    # cv.waitKey(0)

    # 调试
    # cover = cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/sailboat.ppm',0)
    # ls = IntegerLiftingW.from_name('haar')
    # cover = cover.astype(np.int32)
    # ca1, ch1, cv1, cd1 = ls.lwt2(cover)
    # cip = ls.ilwt2(ca1, ch1, cv1, cd1)

    # CAs, CHs, CVs, CDs = [], [], [], []
    # for i in range(3):
    #     cac, chc, cvc, cdc = ls.lwt2(cover[:, :, i])
    #     CAs.append(cac)
    #     CHs.append(chc)
    #     CVs.append(cvc)
    #     CDs.append(cdc)
    # ca1 = np.stack(CAs, axis=2)
    # ch1 = np.stack(CHs, axis=2)
    # cv1 = np.stack(CVs, axis=2)
    # cd1 = np.stack(CDs, axis=2)
    #
    # CIPs = []
    # for i in range(3):
    #     CIPC = ls.ilwt2(ca1[:, :, i], ch1[:, :, i], cv1[:, :, i], cd1[:, :, i])
    #     CIPs.append(CIPC)
    # cip = np.stack(CIPs, axis=2)

    # print(np.sum(abs(cover-cip)))

    from scipy.fftpack import dct, idct
    im = cv.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/avion.ppm', 0)
    dct_im = dct


















