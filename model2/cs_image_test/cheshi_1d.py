# 一维

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
# from date.dataset import get_data_loaders, get_test_data
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import cv2
import math
from collections import OrderedDict
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from torch.optim.lr_scheduler import MultiStepLR
import warnings
warnings.filterwarnings("ignore")
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 网络与参数加载加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

cr = 0.5

from cs_1d2_dct_fil2 import model
path = str(cr)
model = model(in_channels=1,embed_dim=[64, 128, 256, 256, 128, 64], depth=[5, 5, 5, 5, 5, 5]).to(device)

checkpoint_model = path +"/model_epoch_best3.pth"

checkpoint = torch.load(checkpoint_model)
new_state_dict = {}
for k, v in checkpoint.items():
    if k.startswith('_orig_mod.module.'):
        new_key = k[len('_orig_mod.module.'):]
    elif k.startswith('module.'):
        new_key = k[len('module.'):]
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)


# # 数据加载
images_path = 'G:/date/cs_data/Set11'
# images_path = 'G:/date/cs_data/Urban100'

img_paths = os.path.join(images_path)
img_names = os.listdir(img_paths)
# 数据加载
psnr = []
psnr2 = []
ssim = []


date_name = images_path.split('/')[-1]
print('date_name:',date_name)
rim_path = os.path.join('rim',path, date_name)
os.makedirs(rim_path, exist_ok=True)

im_path = os.path.join('im', date_name)
os.makedirs(im_path, exist_ok=True)

# if not os.path.exists(rim_path):
#     os.makedirs(rim_path)

for j in range(len(img_names)):
    print(j)
    img = cv2.imread(images_path + '/' + img_names[j], 0)

    # 裁剪到对应目标大小
    # 获取图像尺寸
    height, width = img.shape[:2]  # 因为读取的是灰度图（参数0），所以shape只有两个维度

    # 剪切到256*256
    set_h = 256
    set_w = 256
    if width > set_h and height > set_w:
        # 计算中心裁剪区域
        left = (width - set_w) // 2
        top = (height - set_h) // 2
        right = left + set_w
        bottom = top + set_h

        # 执行裁剪
        img = img[top:bottom, left:right]
        # print(f"已中心裁剪为 256x256，原尺寸: {width}x{height}")

    img = torch.tensor(img / 255, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    [b, c, h, w] = img.shape

    # 1d置乱矩阵生成方式
    ran = torch.ones([b, c * h * w], dtype=torch.int64)
    ran_inv = torch.ones([b, c * h * w], dtype=torch.int64)
    for i in range(b):
        perm = torch.randperm(c * h * w)  # 生成0~new_h * new_w的数，并在随机置乱。
        perm_inv = torch.empty_like(perm)  # 生成一个大小与perm相同的矩阵
        perm_inv[perm] = torch.arange(perm.shape[0])  # 生成一个perm逆的矩阵。
        ran[i, :] = perm
        ran_inv[i, :] = perm_inv

    cr2 = cr
    # 采样矩阵
    U, S, V = torch.linalg.svd(torch.randn(c * 1024, c * 1024))
    Phi = (U @ V)[:, :c * int(1024 * cr2)]  # @矩阵相乘
    Phit = torch.pinverse(Phi)
    # Phit = Phi.T
    Phi = Phi.repeat(b, 1, 1)
    Phit = Phit.repeat(b, 1, 1)

    if torch.cuda.is_available():
        img = img.to(device)
        Phi = Phi.to(device)
        Phit = Phit.to(device)
        ran = ran.to(device)
        ran_inv = ran_inv.to(device)

    with torch.no_grad():
        rim3, rim = model(img, Phi, Phit, ran, ran_inv)

    img = torch.clip(img[0][0], 0, 1).cpu().detach().numpy()
    rim = torch.clip(rim[0][0], 0, 1).cpu().detach().numpy()
    rim2 = torch.clip(rim3[0][0], 0, 1).cpu().detach().numpy()

    if PSNR(img, rim) > 10:
        psnr_img = PSNR(img, rim)
        psnr_img2 = PSNR(img, rim2)
        ssim_img = SSIM(img, rim)
        psnr.append(psnr_img)
        psnr2.append(psnr_img2)
        ssim.append(ssim_img)
    else:
        print(img_names[j])

    rim_name = str(round(psnr_img, 4))+'_'+str(round(ssim_img, 4))+"_"+img_names[j]
    cv2.imwrite(rim_path+"/"+rim_name,np.uint8(rim*255))
    cv2.imwrite(im_path + "/" + img_names[j], np.uint8(img*255))

# print(psnr)
# print(psnr2)
print(psnr)
print(ssim)
print( 'psnr:', np.mean(np.array(psnr,dtype=float)), 'ssim:', np.mean(np.array(ssim,dtype=float)), 'psnr2:', np.mean(np.array(psnr2,dtype=float)) )


