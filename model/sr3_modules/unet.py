import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from model.sr3_modules import common
import matplotlib.pylab as plt
import numpy as np
import os
import torch.nn.functional as F

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)
        self.c_func =  nn.Conv2d(dim_out, dim_out, 1)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, c):
        # b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)

        h = self.c_func(c) + h
        return h + self.res_conv(x)

 

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input, t=None, save_flag=None, file_num=None):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

 



class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False, size=256):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb, c, t=0, save_flag=False, file_i=0):
        x = self.res_block(x, time_emb, c)   # resblock(x + self.noise_func(noise_embed)) + con1_1(c)
        if(self.with_attn):
            x = self.attn(x, t=t, save_flag=save_flag, file_num=file_i)
        return x


# from scipy.interpolate import interp2d
import cv2
class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        lowres_cond=True,
        condition_ch=3
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

 


        self.res_blocks = res_blocks
        num_mults = len(channel_mults)
        self.num_mults = num_mults
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn,size=now_res))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True,size=now_res),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False,size=now_res)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn, size=now_res))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        
        self.condition = CPEN(inchannel = condition_ch)  # canny+sar
        self.condition_ch = condition_ch
        # self.c_func2 = nn.Linear(128, 128)   #128  256  512  1024
        self.mi = 0




        
    def forward(self, x, time, img_s1=None, class_label=None, return_condition=False, t_ori=0):
        # x torch.cat([x_in['SR'], x_noisy], dim=1)
        condition = x[:, :self.condition_ch, ...].clone()  
        x = x[:, self.condition_ch:, ...]
        

        c1, c2, c3, c4, c5 = self.condition(condition)
        c_base = [c1, c2, c3, c4, c5]
        
 

        
        c = []
        for i in range(len(c_base)):
            for _ in range(self.res_blocks):
                c.append(c_base[i])       

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

 
        
        feats = []
        i=0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                
                x = layer(x, t, c[i])
                # print(x.shape)
                i+=1
            else:
                x = layer(x)

            feats.append(x)
            
            

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t, c5)
                # print(x.shape)
            else:
                x = layer(x)
            

        
        c_base = [c5, c4, c3, c2, c1]
        c = []
        for i in range(len(c_base)):
            for _ in range(self.res_blocks+1):
                c.append(c_base[i])   
        i = 0
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                # print(x.shape)
                x = layer(torch.cat((x, feats.pop()), dim=1), t, c[i])
                # print(x.shape)
                i+=1
            else:
                x = layer(x)
            
        if not return_condition:
            return self.final_conv(x)
        else:
            return self.final_conv(x), [c1, c2, c3, c4, c5]

 

class ResBlock_normal(nn.Module):
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()
        
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


from SoftPool import soft_pool2d, SoftPool2d
class CPEN(nn.Module):
    def __init__(self, inchannel = 1):
        super(CPEN, self).__init__()
        self.pool = SoftPool2d(kernel_size=(2,2), stride=(2,2))
        # self.scale=scale
        # if scale == 2:

        self.E1= nn.Sequential(nn.Conv2d(inchannel, 64, kernel_size=3, padding=1),
                Swish())
        
         

        self.E2=nn.Sequential(
            ResBlock_normal(64, 128, dropout=0, norm_groups=16), 
            ResBlock_normal(128, 128, dropout=0, norm_groups=16),             
            )

        self.E3=nn.Sequential(
            ResBlock_normal(128, 256, dropout=0, norm_groups=16), 
            ResBlock_normal(256, 256, dropout=0, norm_groups=16),             
            )

        self.E4=nn.Sequential(
            ResBlock_normal(256, 512, dropout=0, norm_groups=16), 
            ResBlock_normal(512, 512, dropout=0, norm_groups=16),             
            )

        self.E5=nn.Sequential(
            ResBlock_normal(512, 512, dropout=0, norm_groups=16), 
            ResBlock_normal(512, 1024, dropout=0, norm_groups=16),             
            )

 

    def forward(self, x):

        x1 = self.E1(x)

        x2 = self.pool(x1)
        x2 = self.E2(x2)

        x3 = self.pool(x2)
        x3 = self.E3(x3)


        x4 = self.pool(x3)
        x4 = self.E4(x4)

        x5 = self.pool(x4)
        x5 = self.E5(x5)

        # x5 = self.pool(x5)
        if torch.isnan(x1).any():
            print('x1 nan:\n')
            print(x1)
            print(x2)
            print(x3)
            print(x4)
            print(x5)
        if torch.isnan(x2).any():
            print('x2 nan:\n')
            print(x1)
            print(x2)
            print(x3)
            print(x4)
            print(x5)
        if torch.isnan(x3).any():
            print('x3 nan:\n')
            print(x1)
            print(x2)
            print(x3)
            print(x4)
            print(x5)
        if torch.isnan(x4).any():
            print('x4 nan:\n')
            print(x1)
            print(x2)
            print(x3)
            print(x4)
            print(x5)
        if torch.isnan(x5).any():
            print('x5 nan:\n')
            print(x.mean(),x.min(),x.max())
            print(x1.mean(),x1.min(),x1.max())
            print(x2.mean(),x2.min(),x2.max())
            print(x3.mean(),x3.min(),x3.max())
            print(x4.mean(),x4.min(),x4.max())
            print(x5.mean(),x5.min(),x5.max())
            for k, v  in self.E1.named_parameters():
                print(k, ':\n', v.mean(),v.min(),v.max())
            for k, v  in self.E2.named_parameters():
                print(k, ':\n', v.mean(),v.min(),v.max())
            for k, v  in self.E3.named_parameters():
                print(k, ':\n', v.mean(),v.min(),v.max())            
            for k, v  in self.E4.named_parameters():
                print(k, ':\n', v.mean(),v.min(),v.max())           
            for k, v  in self.E5.named_parameters():
                print(k, ':\n', v.mean(),v.min(),v.max())
        return x1, x2, x3, x4, x5
