# encoder采用fft，dencoder采用wt
# 都添加测量值残差
# 一维分块采样
# 直接在cs_1d2_dct上加卷积滤波

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


def scrambling_sampling_1d(img,phi,perm):
    # 采用小波变换进行稀疏化
    [b, c, h, w] = img.shape
    N = 1024 * c  # 每块数的个数，彩色为1024*3
    # 稀疏
    dct_img = dct.dct_2d(img)
    # 置乱
    scr_img = torch.ones(perm.shape,device=img.device)
    for i in range(b):
        scr_img[i,:] = dct_img[i,:].reshape(-1, )[perm[i,:]]
    scr_img = scr_img.reshape(b,-1, N)
    # 采样
    sap_img = scr_img @ phi
    return sap_img

def Reconstruct_rscrambling_1d(img,Phit,perm_inv,size):

    [b,c,h,w] = size
    # 重构
    Reconstruct_img = img @ Phit
    # 反向置乱
    rscrambling_img = torch.ones([b,c*h*w],device=img.device)
    for i in range(b):
        rscrambling_img[i,:] = Reconstruct_img[i,:].reshape(-1, )[perm_inv[i,:]]
    rscrambling_img = rscrambling_img.reshape(b,c,h,w)
    # 稠密化
    density_img = dct.idct_2d(rscrambling_img)

    return density_img

class SS(nn.Module):
    def __init__(self, in_chan, mid_chan):
        super().__init__()
        self.head = nn.Conv2d(in_chan, mid_chan, 3, padding=1)
        self.body = nn.Sequential(*[nn.Conv2d(mid_chan, mid_chan,3,1,1) for _ in range(5)])
        self.tail = nn.Conv2d(mid_chan, in_chan, 3, padding=1)

    def forward(self, x):
        return self.tail(self.body(self.head(x)))

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

        # 计算注意力
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.einsum('bci,bcj->bij', q, k) * (c ** (-0.5))
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.reshape(b, c, h, w)

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
                       groups=dim * 4) for _ in range(self.num)]
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

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
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
        self.ss = SS(1, 32)
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



if __name__ == '__main__':
    import cv2
    from skimage.metrics import structural_similarity as SSIM
    from skimage.metrics import peak_signal_noise_ratio as PSNR

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # img = cv2.imread('H:\coder\cs\Random_sampling_matrix_compression\date\parrots.tif')
    # img = torch.tensor(img / 255, dtype=torch.float32).unsqueeze(0).to(device).permute(0,3,1,2)

    img = cv2.imread('G:\date\cs_data\ccia_CVG_image\color_image_512/avion.ppm',0)
    img = np.double(cv2.resize(img, [1024, 1024]))
    img = torch.tensor(img / 255, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    [b, c, h, w] = img.shape

    cr=0.25
    cr1=cr
    # 1d置乱矩阵生成方式
    ran = torch.ones([b, c * h * w], dtype=torch.int64)
    ran_inv = torch.ones([b, c * h * w], dtype=torch.int64)
    for i in range(b):
        perm = torch.randperm(c * h * w)  # 生成0~new_h * new_w的数，并在随机置乱。
        perm_inv = torch.empty_like(perm)  # 生成一个大小与perm相同的矩阵
        perm_inv[perm] = torch.arange(perm.shape[0])  # 生成一个perm逆的矩阵。
        ran[i, :] = perm
        ran_inv[i, :] = perm_inv

    # 采样矩阵
    U, S, V = torch.linalg.svd(torch.randn(c * 1024, c * 1024))
    Phi = (U @ V)[:, :c * int(1024*cr1)]  # @矩阵相乘
    Phit = torch.pinverse(Phi)
    # Phit = Phi.T
    Phi = Phi.repeat(b, 1, 1)
    Phit = Phit.repeat(b, 1, 1)


    Phi = Phi.to(device)
    Phit = Phit.to(device)
    ran = ran.to(device)
    ran_inv = ran_inv.to(device)


    module = model(c,embed_dim=[64, 128, 256, 256, 128, 64], depth=[5, 5, 5, 5, 5, 5]).to(device)

    checkpoint_model = torch.load('model_epoch_best2.pth')

    # 创建新的 state_dict，去掉 _orig_mod. 前缀
    new_state_dict = {}
    for k, v in checkpoint_model.items():
        # 去掉前缀 _orig_mod.
        if k.startswith('_orig_mod.module'):
            new_key = k[len('_orig_mod.module.'):]  # 去掉前缀
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # 加载处理后的权重
    module.load_state_dict(new_state_dict)

    print('模型加载成功')
    del checkpoint_model
    del new_state_dict

    with torch.no_grad():
        size = img.shape
        [b,c,m,n] = img.shape
        # out30, rim = module(img, Phi, Phit, ran, ran_inv)
        Y = module.comp(img, Phi, ran)

        # max_x2 = torch.max(Y)
        # min_x2 = torch.min(Y)
        # Y = torch.round( ((Y - min_x2) / (max_x2 - min_x2)) * 255 )
        # Y = Y / 255 * (max_x2 - min_x2) + min_x2

        Y = Y.cpu()
        max_x2 = torch.max(Y)
        min_x2 = torch.min(Y)
        x1 = np.array(torch.round((Y - min_x2) / (max_x2 - min_x2) * 255))
        max_min = [max_x2, min_x2]
        x1 = x1.reshape(int(m / 2), int(n / 2))


        x1 = x1.reshape(-1, 256)
        Y = torch.tensor(x1).unsqueeze(0).float()
        max_x2, min_x2 = max_min[0], max_min[1]
        Y = Y / 255 * (max_x2 - min_x2) + min_x2
        Y = Y.to(device)

        rim = module.refact( Y, Phi, Phit, ran, ran_inv, size)

    img = torch.clip(img[0][0], 0, 1).cpu().detach().numpy()
    rim = torch.clip(rim[0][0], 0, 1).cpu().detach().numpy()
    # out30 = torch.clip(out30[0][0], 0, 1).cpu().detach().numpy()

    print(np.sum(abs(img-rim)))
    print(rim.shape)
    print('psnr:', PSNR(img, rim), 'ssim:', SSIM(np.uint8(img * 255), np.uint8(rim * 255)))

    # 展示
    cv2.imshow('img', (img))
    cv2.imshow('rim', (rim))
    cv2.waitKey(0)







