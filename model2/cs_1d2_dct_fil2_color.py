
# encoder采用fft，dencoder采用wt
# 都添加测量值残差
# 一维分块采样
# 直接在cs_1d2_dct上加卷积滤波
# 彩色版本：每个通道独立压缩，支持各通道按不同比例分配总采样率

import torch
import torch.nn as nn
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import torch.utils.checkpoint as checkpoint
import math
from torch import Tensor
from torchstat import stat
import torch.nn.functional as F
import numbers
import numpy as np

from einops import rearrange

import pywt
import pywt.data
import torch.nn.functional as F
import torch_dct as dct


class Upsample(nn.Module):
    """
    UPSample: Conv
    B*(H/2)*(W/2)*(2*C) -> B*H*W*C
    """
    def __init__(self,input_dim=96, out_dim=64, patch_size=2):
        super().__init__()

        self.input_dim = input_dim

        self.proj = nn.Sequential(  # 2C -> 4C
            nn.Conv2d(input_dim, out_dim * patch_size ** 2, kernel_size=1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = F.pixel_shuffle(x, 2)
        return x

class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """
    def __init__(self, input_dim, output_dim, patch_size=2):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(input_dim * patch_size ** 2, output_dim, kernel_size=1))

    def forward(self, x):
        x = F.pixel_unshuffle(x,2)
        x = self.proj(x)
        return x


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.dim = in_channels

        self.conv_layer = nn.Sequential(
                                        nn.BatchNorm2d(out_channels * 2),
                                        nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                                        kernel_size=1, stride=1, padding=0, groups=self.groups,bias=True),
                                        nn.GELU(),
                                        )

    def forward(self, x):
        batch, c, h, w = x.size()
        # (batch, c, h, w/2+1, 2)
        # ffted = torch.fft.fftshift(torch.fft.rfft2(x, norm='ortho'), dim=(-2,-1))  ################
        X_fft = torch.fft.fft2(x, dim=(-2, -1))
        ffted = torch.fft.fftshift(X_fft, dim=(-2, -1))

        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)

        X_fft_unshifted = torch.fft.ifftshift(ffted, dim=(-2, -1))
        X_recon_complex = torch.fft.ifft2(X_fft_unshifted, dim=(-2, -1))
        X_recon = torch.real(X_recon_complex)

        # output = torch.fft.irfft2(torch.fft.ifftshift(ffted, dim=(-2,-1)), s=(h, w), norm='ortho')   ###################

        return X_recon

class OurTokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(OurTokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.FFC(x)
        x = self.conv_fina(x)

        return x


class OurTokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(OurTokenMixer_For_Local, self).__init__()
        self.dim = dim

        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.dim),
            nn.Conv2d(self.dim, self.dim*2, kernel_size=3, padding=1),
            nn.GELU(),

            nn.BatchNorm2d(self.dim*2),
            nn.Conv2d(self.dim*2, self.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def scrambling_sampling_1d(img, phi, perm):
    """
    彩色/多通道分通道独立一维分块采样。

    参数:
        img:  [b, c, h, w]
        phi:  list of c 个采样矩阵，每个为 [b, 1024, m_i]
              （m_i 为第 i 通道每块测量数）
        perm: list of c 个置乱序列，每个为 [b, h*w]

    返回:
        sap_img: [b, num_blocks, M]，M = sum(m_i)
    """
    [b, c, h, w] = img.shape
    num_blocks = h * w // 1024
    # 稀疏
    dct_img = dct.dct_2d(img)
    sap_list = []
    for i in range(c):
        # 取出第 i 通道
        chan_flat = dct_img[:, i, :, :].reshape(b, h * w)  # [b, h*w]
        # 向量化置乱：使用 gather 替代 batch 内逐样本循环
        perm_i = perm[i]                                   # [b, h*w]
        scr_img = torch.gather(chan_flat, 1, perm_i)       # [b, h*w]
        scr_img = scr_img.reshape(b, num_blocks, 1024)     # [b, num_blocks, 1024]
        # 采样
        sap_list.append(scr_img @ phi[i])                  # [b, num_blocks, m_i]
    sap_img = torch.cat(sap_list, dim=-1)                  # [b, num_blocks, M]
    return sap_img


def Reconstruct_rscrambling_1d(img, Phit, perm_inv, size):
    """
    彩色/多通道分通道独立重构。

    参数:
        img:       [b, num_blocks, M]
        Phit:      list of c 个逆采样矩阵，每个为 [b, m_i, 1024]
        perm_inv:  list of c 个逆置乱序列，每个为 [b, h*w]
        size:      [b, c, h, w]

    返回:
        density_img: [b, c, h, w]
    """
    [b, c, h, w] = size
    num_blocks = h * w // 1024
    # 按通道切分测量值
    m_list = [Phit[i].shape[1] for i in range(c)]
    sap_splits = torch.split(img, m_list, dim=-1)
    recon_list = []
    for i in range(c):
        # 重构
        Reconstruct_img = sap_splits[i] @ Phit[i]       # [b, num_blocks, 1024]
        Reconstruct_img = Reconstruct_img.reshape(b, h * w)
        # 向量化反向置乱：使用 gather 替代 batch 内逐样本循环
        perm_inv_i = perm_inv[i]
        rscrambling_img = torch.gather(Reconstruct_img, 1, perm_inv_i)  # [b, h*w]
        rscrambling_img = rscrambling_img.reshape(b, h, w)
        recon_list.append(rscrambling_img)
    # 合并通道并稠密化
    density_img = torch.stack(recon_list, dim=1)        # [b, c, h, w]
    density_img = dct.idct_2d(density_img)
    return density_img

class SS(nn.Module):
    def __init__(self, in_chan, mid_chan):
        super().__init__()
        self.head = nn.Conv2d(in_chan, mid_chan, 3, padding=1)
        self.body = nn.Sequential(*[nn.Conv2d(mid_chan, mid_chan,3,1,1) for _ in range(5)])
        self.tail = nn.Conv2d(mid_chan, in_chan, 3, padding=1)

    def forward(self, x):
        return self.tail(self.body(self.head(x)))

class SSPerChannel(nn.Module):
    """
    每个通道独立的 SS 预处理网络。
    输入/输出通道数均为 num_channels，内部对每个通道单独过 SS(1, mid_chan)。
    """
    def __init__(self, num_channels, mid_chan):
        super().__init__()
        self.num_channels = num_channels
        self.ss_list = nn.ModuleList([SS(1, mid_chan) for _ in range(num_channels)])

    def forward(self, x):
        # x: [b, num_channels, h, w]
        out = [self.ss_list[i](x[:, i:i+1, :, :]) for i in range(self.num_channels)]
        return torch.cat(out, dim=1)

class com_recon2(nn.Module):
    # 每个模块添加的压缩重构部分
    # 输入维度dim与当前与原图间的大小差距cr
    def __init__(
            self, dim, cr, chan
    ):
        super(com_recon2, self).__init__()
        self.dim = dim
        self.cr = cr
        self.conv1 = nn.Conv2d(int(self.dim/(self.cr**2)), chan, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2*chan, int(self.dim/(self.cr**2)), kernel_size=3, padding=1)

    def forward(self, x, y1, phi, Phit, perm, perm_inv, ss):
        copy = x
        x = F.pixel_shuffle(x,self.cr)
        x = self.conv1(x)

        x1d = ss(x)
        y1d = scrambling_sampling_1d(x1d,phi,perm)
        x1d = Reconstruct_rscrambling_1d(y1 - y1d,Phit,perm_inv,x.shape)

        x = torch.cat([x,x1d],1)

        x = self.conv2(x)
        x = F.pixel_unshuffle(x,self.cr) + copy

        return x

class OurBlock(nn.Module):
    def __init__(
            self,
            dim,
            cr,
            chan,
            Gloal=OurTokenMixer_For_Gloal,
            Local=OurTokenMixer_For_Local,
            comrecon = com_recon2   ###########################################################################这个地方改
    ):
        super(OurBlock, self).__init__()
        self.dim = dim
        self.cr = cr
        self.chan = chan

        self.conv_init = nn.Sequential(  # PW->DW->
            nn.BatchNorm2d(self.dim),
            nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1),
        )

        self.gloal = Gloal(dim)
        self.local = Local(dim)

        self.conv_out = nn.Sequential(  # PW->DW->
            nn.Conv2d(2*dim, dim, 1),
        )

        self.com_recon = comrecon(dim=self.dim,cr=self.cr,chan=self.chan)

    def forward(self, x, y1, phi, Phit, perm, perm_inv,ss):
        copx = x

        x = self.conv_init(x)
        x_g,x_l = torch.chunk(x, chunks=2, dim=1)
        xg = self.gloal(x_g)
        xl = self.local(x_l)
        x = torch.cat((xg,xl),1)
        x = self.conv_out(x)
        x = x + copx

        x = self.com_recon(x, y1, phi, Phit, perm, perm_inv,ss)

        return x


class OurStage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
            cr=int,
            chan=int,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(OurStage, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
                OurBlock(
                    dim=in_channels,
                    cr=cr,
                    chan=chan
                )
            for index in range(depth)
        ])

    def forward(self, x, y1, phi, Phit, perm, perm_inv,ss):
        for block in self.blocks:
            x = block(x, y1, phi, Phit, perm, perm_inv,ss)
        return x


class enc(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(enc, self).__init__()

        self.sp = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 5, 1, 2),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.sp(x)
        return x

class dec(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(dec, self).__init__()

        self.sp = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, out_dim, 5, 1, 2),
        )

    def forward(self, x):
        x = self.sp(x)
        return x



def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    ####print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros(
        [out_batch, out_channel, out_height, out_width],
        device=x.device,  # 使用与 X 相同的设备（CPU/GPU）
        dtype=x.dtype  # 使用与 X 相同的数据类型
    )

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        a, b = dwt_init(x)
        return a, b

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class FAM(nn.Module):
    def __init__(self, int_channel, out_channel):
        super(FAM, self).__init__()
        self.num = int((int_channel/out_channel)/2)  #计算其差几倍
        self.wt = DWT()
        self.iwt = IWT()

        self.merge = nn.Sequential(
            nn.Conv2d(int_channel+out_channel, int_channel+out_channel, 3, 1, 1, groups=int_channel+out_channel),
            nn.GELU(),
            nn.Conv2d(int_channel+out_channel, out_channel, 1),
        )

        self.img_convs = nn.ModuleList(
            [nn.Conv2d(out_channel*4, out_channel*4, 3, padding=1, stride=1, dilation=1,
                       groups=out_channel*4, bias=False) for _ in range(self.num)]
        )

    def forward(self, x1, x2):
        """
        x1:低维的与高维的x2在ll通道融合
        """

        x_h_in_levels = []
        ll = x2.clone()
        for i in range(self.num):
            ll,h = self.wt(ll)

            x_h_in_levels.append(h)

        ll = self.merge(torch.cat([x1, ll], dim=1))

        for i in range(self.num - 1, -1, -1):
            # 从列表中取出对应级别的低频和高频子带
            h = x_h_in_levels.pop()
            ll = self.img_convs[i](torch.cat([ll, h], dim=1))
            ll = self.iwt(ll)

        out = ll + x2

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 计算注意力，使用 bmm 替代 einsum，通常更省显存/更快
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.bmm(q.transpose(1, 2), k) * (c ** (-0.5))  # [b, hw, hw]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v.transpose(1, 2))  # [b, hw, c]
        out = out.transpose(1, 2).reshape(b, c, h, w)

        return x + self.proj_out(out)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, skip_channels=0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels + skip_channels  # 考虑跳跃连接的额外通道

        self.norm1 = nn.GroupNorm(32, self.in_channels)
        self.conv1 = nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if self.in_channels != out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

class wt_ll(nn.Module):
    # 高维采用atention聚焦全局
    def __init__(self, dim, num):

        super(wt_ll, self).__init__()

        self.num = num
        self.wt = DWT()
        self.iwt = IWT()

        self.dim = dim
        self.att = SelfAttention(self.dim)
        self.Res = ResBlock(self.dim)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(dim*4, dim*4, 3, padding=1, stride=1, dilation=1,
                       groups=dim*4) for _ in range(self.num)]
        )

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(dim * 4, dim * 4, 3, padding=1, stride=1, dilation=1,
                       groups=dim*4) for _ in range(self.num)]
        )

    def forward(self, x):

        x_h_in_levels = []
        ll = x.clone()
        for i in range(self.num):
            ll, h = self.wt(ll)
            tem = self.convs1[i](torch.cat([ll, h], dim=1))
            split_x = torch.tensor_split(tem, [self.dim], dim=1)
            ll, h = split_x[0], split_x[1]
            x_h_in_levels.append(h)

        ll = self.Res(self.att(ll))

        for i in range(self.num - 1, -1, -1):
            # 从列表中取出对应级别的低频和高频子带
            h = x_h_in_levels.pop()
            ll = self.convs2[i](torch.cat([ll, h], dim=1))
            ll = self.iwt(ll)

        return ll+x

class LayerNorm2d(nn.Module):
    # 使用 PyTorch 原生 layer_norm 替换手写版 autograd Function，保持参数名 weight/bias 不变
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x: [b, c, h, w]
        x = x.permute(0, 2, 3, 1)  # [b, h, w, c]
        x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)  # [b, c, h, w]
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)

class DBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=[1,2,4], extra_depth_wise=True):

        super(DBlock, self).__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1, padding=0, stride=1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.branches = nn.ModuleList()  # 膨胀卷积
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def forward(self, inp):

        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
        x = self.conv4(self.norm2(y))  # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x)  # size [B, C, H, W]
        x = y + x * self.gamma

        return x

class wt_h(nn.Module):
    "小波高频进行多尺度"
    def __init__(self,dim):

        super(wt_h, self).__init__()
        self.dim = dim
        self.DBlock=DBlock(dim)

    def forward(self, x):
        x = self.DBlock(x)
        return x

class OurBlock_den(nn.Module):
    def __init__(
            self,
            dim,
            cr,
            num,
            chan,
            wt_ll=wt_ll,
            wt_h=wt_h,
            comrecon=com_recon2
    ):
        super(OurBlock_den, self).__init__()
        self.dim = dim
        self.cr = cr
        self.num = num
        self.chan = chan

        self.wt = DWT()
        self.iwt = IWT()

        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
        )

        self.gloal = wt_ll(dim, num)
        self.local = wt_h(dim*3)

        self.conv_out = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim, 3,1,1),
        )

        self.com_recon = comrecon(dim=self.dim, cr=self.cr, chan=self.chan)


    def forward(self, x, y1, phi, Phit, perm, perm_inv, ss):

        copx = x
        x = self.conv_init(x)
        ll,h = self.wt(x)
        ll = self.gloal(ll)
        h = self.local(h)
        x = self.iwt(torch.cat((ll,h),1))
        x = self.conv_out(x)
        x = x + copx

        x = self.com_recon(x, y1, phi, Phit, perm, perm_inv, ss)

        return x

class OurStage_den(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
            num=int,
            cr=int,
            chan=int,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(OurStage_den, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
                OurBlock_den(
                    dim=in_channels,
                    cr=cr,
                    num=num,
                    chan=chan,
                )
            for index in range(depth)
        ])

    def forward(self, x, y1, phi, Phit, perm, perm_inv,ss):
        for block in self.blocks:
            x = block(x, y1, phi, Phit, perm, perm_inv,ss)
        return x

class moder_encoder(nn.Module):
    def __init__(self, in_chans=3,
                 embed_dim=[64, 128, 256], depth=[3, 3, 3], cr=[1, 2, 4]):
        super(moder_encoder, self).__init__()

        self.encoder = enc(in_chans, embed_dim[0])

        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1])
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2])

        self.layer1 = OurStage(depth=depth[0], in_channels=embed_dim[0], cr=cr[0], chan=in_chans)
        self.layer2 = OurStage(depth=depth[1], in_channels=embed_dim[1], cr=cr[1], chan=in_chans)
        self.layer3 = OurStage(depth=depth[2], in_channels=embed_dim[2], cr=cr[2], chan=in_chans)

        self.decoder30 = dec(embed_dim[2], in_chans * 16)


    def forward(self, x1, y1, phi, Phit, perm, perm_inv,ss):

        x = self.encoder(x1)
        x = self.layer1(x, y1, phi, Phit, perm, perm_inv,ss)
        # copy1 = x

        x = self.downsample1(x)
        x = self.layer2(x, y1, phi, Phit, perm, perm_inv,ss)
        # copy2 = x

        x = self.downsample2(x)
        x = self.layer3(x, y1, phi, Phit, perm, perm_inv,ss)

        out30 = self.decoder30(x)

        return x, F.pixel_shuffle(out30,4)

class moder_dencoder(nn.Module):
    """
    cr=[4, 2, 1]为下采样个数
    num=[1, 2, 3]为低频部分小波变换次数
    """
    def __init__(self, in_chans=3,
                 embed_dim=[256, 128, 64], depth=[3, 3, 3], num=[1, 2, 3], cr=[4, 2, 1]):
        super(moder_dencoder, self).__init__()

        out_chans = in_chans

        self.upsample2 = Upsample(input_dim=embed_dim[0], out_dim=embed_dim[1])
        self.upsample1 = Upsample(input_dim=embed_dim[1], out_dim=embed_dim[2])

        self.layer3 = OurStage_den(depth=depth[0], in_channels=embed_dim[0], num=num[0], cr=cr[0], chan=in_chans)
        self.layer4 = OurStage_den(depth=depth[1], in_channels=embed_dim[1], num=num[1], cr=cr[1], chan=in_chans)
        self.layer5 = OurStage_den(depth=depth[2], in_channels=embed_dim[2], num=num[2], cr=cr[2], chan=in_chans)

        self.skip1 = FAM(embed_dim[0],embed_dim[1])
        self.skip2 = FAM(embed_dim[0],embed_dim[2])

        self.decoder1 = dec(embed_dim[2], out_chans)

    def forward(self, copy3, y1, phi, Phit, perm, perm_inv,ss):

        x = self.layer3(copy3, y1, phi, Phit, perm, perm_inv,ss)

        x = self.upsample2(x)
        x = self.skip1(copy3, x)
        x = self.layer4(x, y1, phi, Phit, perm, perm_inv,ss)

        x = self.upsample1(x)
        x = self.skip2(copy3, x)
        x = self.layer5(x, y1, phi, Phit, perm, perm_inv,ss)
        out = self.decoder1(x)

        return out


class model(nn.Module):

    def __init__(self, in_channels, embed_dim=[64, 128, 256, 256, 128, 64], depth=[3, 3, 3, 3, 3, 3]):
        super().__init__()

        self.in_channels = in_channels
        self.ss = SSPerChannel(self.in_channels, 32)
        self.encoder = moder_encoder(in_channels, embed_dim[0:3], depth[0:3])
        self.dencoder = moder_dencoder(in_channels, embed_dim[3:], depth[3:])

    def forward(self, x, phi, Phit, perm, perm_inv):
        """
        x:原图
        Phi, PhiT：测量矩阵
        perm, perm_inv：置乱控制序列
        par：k稀疏控制参数，随着压缩率变
        """
        x = self.ss(x)
        y1d = scrambling_sampling_1d(x,phi,perm)
        x1d = Reconstruct_rscrambling_1d(y1d,Phit,perm_inv,x.shape)

        fea, out30, = self.encoder(x1d, y1d, phi, Phit, perm, perm_inv, self.ss)
        out = self.dencoder(fea, y1d, phi, Phit, perm, perm_inv, self.ss)

        return out30, out

    def comp(self, x, phi, perm):
        """
        x:原图
        Phi, PhiT：测量矩阵
        perm, perm_inv：置乱控制序列
        par：k稀疏控制参数，随着压缩率变
        """
        x = self.ss(x)
        y1d = scrambling_sampling_1d(x,phi,perm)
        return y1d

    def refact(self, y1d, phi, Phit, perm, perm_inv, size):
        """
        重构
        """
        x1d = Reconstruct_rscrambling_1d(y1d, Phit, perm_inv, size)
        fea, out30, = self.encoder(x1d, y1d, phi, Phit, perm, perm_inv, self.ss)
        out = self.dencoder(fea, y1d, phi, Phit, perm, perm_inv, self.ss)
        return out



def build_per_channel_measurements(b, c, h, w, cr_list=None, cr_total=None, split_ratios=None, device='cpu'):
    """
    生成彩色/多通道独立压缩所需的测量矩阵与置乱序列。

    参数（二选一）:
        cr_list:      list of c，直接指定每个通道的采样率（相对该通道 1024 块）
        cr_total:     总采样率（测量值总数 / 总像素数）
        split_ratios: list of c，各通道分配总采样率的比例，求和需为 1

    返回:
        phi_list, Phit_list, perm_list, perm_inv_list
    """
    assert h % 32 == 0 and w % 32 == 0, "图像高宽需为 32 的倍数"
    if cr_list is not None:
        assert len(cr_list) == c, "cr_list 长度需等于通道数"
        channel_rates = cr_list
    elif cr_total is not None and split_ratios is not None:
        assert len(split_ratios) == c, "split_ratios 长度需等于通道数"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios 求和需为 1"
        channel_rates = [cr_total * 3.0 * r for r in split_ratios]
    else:
        raise ValueError("请传入 cr_list 或 (cr_total, split_ratios)")

    phi_list = []
    Phit_list = []
    perm_list = []
    perm_inv_list = []
    for i in range(c):
        m_i = max(1, int(1024 * channel_rates[i]))
        # 使用 QR 分解生成正交矩阵，避免昂贵的 SVD；列正交，伪逆即转置
        Q, _ = torch.linalg.qr(torch.randn(1024, 1024))
        phi_i = Q[:, :m_i]            # [1024, m_i]
        Phit_i = phi_i.T              # [m_i, 1024]
        phi_list.append(phi_i.repeat(b, 1, 1).to(device))
        Phit_list.append(Phit_i.repeat(b, 1, 1).to(device))

        # 向量化生成置乱序列，避免逐 batch 循环
        rand = torch.rand(b, h * w)
        perm = torch.argsort(rand, dim=-1)
        perm_inv = torch.argsort(perm, dim=-1)
        perm_list.append(perm.to(device))
        perm_inv_list.append(perm_inv.to(device))

    return phi_list, Phit_list, perm_list, perm_inv_list


if __name__ == '__main__':
    import cv2
    from skimage.metrics import structural_similarity as SSIM
    from skimage.metrics import peak_signal_noise_ratio as PSNR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取彩色图像，需要保证高宽为 32 的倍数
    img_path = r'H:\coder\cs\Random_sampling_matrix_compression\date\parrots.tif'
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        # 若图像不存在，则随机生成一张测试图
        print('未找到测试图像，使用随机 256x256 彩色图进行演示')
        img_bgr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # 裁剪或缩放至 32 的倍数
    h0, w0 = img_bgr.shape[:2]
    h = (h0 // 32) * 32
    w = (w0 // 32) * 32
    img_bgr = cv2.resize(img_bgr, (w, h))

    # BGR -> RGB，归一化并转为 [b, c, h, w]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img_rgb / 255.0, dtype=torch.float32).unsqueeze(0).to(device).permute(0, 3, 1, 2)

    [b, c, h, w] = img.shape

    # 示例：总采样率 0.25，按 [0.4, 0.35, 0.25] 分给 R/G/B
    # 每个通道实际采样率 = cr_total * 3 * split_ratios[i]
    # cr_total = 0.25
    # split_ratios = [0.7, 0.2, 0.1]
    # Phi, Phit, perm, perm_inv = build_per_channel_measurements(
    #     b, c, h, w, cr_total=cr_total, split_ratios=split_ratios, device=device
    # )

    # Y = scrambling_sampling_1d(img, Phi, perm)
    # rim = Reconstruct_rscrambling_1d(Y, Phit, perm_inv, img.shape)

    cr=0.25
    split_ratios = [0.6, 0.2, 0.2]
    channel_rates = [cr * 3.0 * r for r in split_ratios]
    m_all = [int(1024 * r) for r in channel_rates]
    m_all[0] = int(1024 * cr) - (m_all[1] - m_all[2])

    phi_list = []
    Phit_list = []
    perm_list = []
    perm_inv_list = []
    for i in range(c):
        m_i = m_all[i]
        max_retries = 3  # 最多重试2次（加上初次尝试共3次）
        for attempt in range(max_retries + 1):
            try:
                # 使用 QR 分解替代 SVD，列正交伪逆为转置
                Q, _ = torch.linalg.qr(torch.randn(1024, 1024))
                Phi_i = Q[:, :m_i]  # @矩阵相乘
                Phit_i = Phi_i.T
                break  # 成功则跳出重试循环
            except RuntimeError as e:
                if attempt < max_retries:
                    print(f"伪逆计算失败，正在重试... (尝试 {attempt + 1}/{max_retries + 1})")
                    # 可选：打印错误信息以便调试
                    # print(f"错误原因: {e}")
                else:
                    print("重试后仍然失败，请检查矩阵条件数或调整参数。")
                    raise  # 最后一次重试仍失败则抛出异常
        phi_list.append(Phi_i.repeat(b, 1, 1).to(device))
        Phit_list.append(Phit_i.repeat(b, 1, 1).to(device))

        # 向量化生成置乱序列
        rand = torch.rand(b, h * w)
        perm = torch.argsort(rand, dim=-1)
        perm_inv = torch.argsort(perm, dim=-1)
        perm_list.append(perm.to(device))
        perm_inv_list.append(perm_inv.to(device))

    mode = model(c, embed_dim=[32, 64, 128, 128, 64, 32], depth=[3, 3, 3, 3, 3, 3]).to(device)
    with torch.no_grad():
        out30, rim = mode(img, phi_list, Phit_list, perm_list, perm_inv_list)

    img_np = torch.clip(img[0], 0, 1).cpu().detach().numpy().transpose(1, 2, 0)
    rim_np = torch.clip(rim[0], 0, 1).cpu().detach().numpy().transpose(1, 2, 0)
    # out30_np = torch.clip(out30[0], 0, 1).cpu().detach().numpy().transpose(1, 2, 0)

    print('差值和:', np.sum(np.abs(img_np - rim_np)))
    print('重构尺寸:', rim_np.shape)
    print('psnr:', PSNR(img_np, rim_np), 'ssim:', SSIM(
        np.uint8(img_np * 255), np.uint8(rim_np * 255), channel_axis=-1, data_range=255
    ))

    # 展示（OpenCV 使用 BGR）
    cv2.imshow('img', cv2.cvtColor(np.uint8(img_np * 255), cv2.COLOR_RGB2BGR))
    cv2.imshow('rim', cv2.cvtColor(np.uint8(rim_np * 255), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
