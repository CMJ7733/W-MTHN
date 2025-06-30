import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
from base_networks import *
from einops import rearrange 

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
import pdb
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse

class WaveletPreprocessor(nn.Module):
    def __init__(self, wave='sym2', levels=3, enhance_factor=[1.1, 1.2, 1.3]):
        """
        小波预处理模块, 使用symlet小波变换增强图像细节
        
        参数:
            wave(str): 使用的小波名称
            enhance_factor (float): 高频细节增强因子
        """
        super().__init__()
        self.wave = wave
        self.levels = levels
        self.enhance_factor = enhance_factor
        self.DWT = DWTForward(J=levels, wave='sym2', mode='symmetric') 
        self.idwt = DWTInverse(wave=wave, mode='symmetric')
    
    def forward(self, x):
        """
        输入: 
            x (tensor): 输入图像 [B, 3, H, W]
        
        返回:
            enhanced (tensor): 增强后的图像 [B, 3, H, W]
        """
        with torch.cuda.amp.autocast(enabled=False):
            original_dtype = x.dtype
            x = x.float()
            # 小波分解
            yl, yh = self.DWT(x)
            high_freq = yh[0] 
            
            enhanced_yh = []
            for i in range(self.levels):
                high_freq = yh[i]
                enhanced_high = high_freq * self.enhance_factor[i]
                enhanced_yh.append(enhanced_high)
            
            # 重组图像
            enhanced = self.idwt((yl, enhanced_yh))
            
            # 融合原始图像和增强后的图像
            fused = (x + enhanced) / 2
            return fused.to(original_dtype)

class WaveletFeatureFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, 3),  # 3个方向
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 1)  # 修改输出通道数
        self.DWT = DWTForward(J=1, wave='sym2', mode='symmetric')
        
        # 添加上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            original_dtype = x.dtype
            x = x.float()
            B, C, H, W = x.size()  # 获取输入张量的尺寸
            
            # 小波分解
            yl, yh = self.DWT(x)
            LH, HL, HH = yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]
            
            # 注意力权重
            weights = self.fc(self.avg_pool(x).view(B, C))
            w_LH, w_HL, w_HH = weights.chunk(3, dim=1)
            w_LH = w_LH.view(B, 1, 1, 1)  
            w_HL = w_HL.view(B, 1, 1, 1)
            w_HH = w_HH.view(B, 1, 1, 1)
            
            # 加权融合 (正确广播)
            wavelet_features = w_LH * LH + w_HL * HL + w_HH * HH
            
            # 先上采样
            wavelet_features = self.upsample(wavelet_features)
            
            # 然后再进行精确尺寸对齐
            wavelet_features = F.interpolate(
                wavelet_features, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 与原始特征融合
            fused = torch.cat([x, wavelet_features], dim=1)
            return self.conv(fused).to(original_dtype)

class EncoderTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], 
                 use_mamba=True, mamba_chunk_size=64, mamba_parallel=True, 
                 use_mamba_stages=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        
        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        # for Intra-patch transformer blocks

        self.mini_patch_embed1 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                          embed_dim=embed_dims[1])
        self.mini_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.mini_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.mini_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 32, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[3])
        # DynamicCrossAttention
        self.cross_attn1 = DynamicCrossAttention(embed_dims[0], embed_dims[1])
        self.cross_attn2 = DynamicCrossAttention(embed_dims[1], embed_dims[2])
        self.cross_attn3 = DynamicCrossAttention(embed_dims[2], embed_dims[3])
        self.cross_attn4 = DynamicCrossAttention(embed_dims[3], embed_dims[3])
        
        # mamba_stages
        if use_mamba_stages is None:
            use_mamba_stages = [use_mamba] * len(depths)
        else:
            assert len(use_mamba_stages) == len(depths), \
                f"use_mamba_stages length ({len(use_mamba_stages)}) must match depths length ({len(depths)})"

        # main  encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0],
            use_mamba=use_mamba_stages[0],
            mamba_chunk_size=mamba_chunk_size,
            mamba_parallel=mamba_parallel)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        # intra-patch encoder
        self.patch_block1 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(1)])
        self.pnorm1 = norm_layer(embed_dims[1])
        # main  encoder
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1],
            use_mamba=use_mamba_stages[1],
            mamba_chunk_size=mamba_chunk_size,
            mamba_parallel=mamba_parallel)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        # intra-patch encoder
        self.patch_block2 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1)])
        self.pnorm2 = norm_layer(embed_dims[2])
        # main  encoder
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2],
            use_mamba=use_mamba_stages[2],
            mamba_chunk_size=mamba_chunk_size,
            mamba_parallel=mamba_parallel)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        # intra-patch encoder
        self.patch_block3 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[1], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1)])
        self.pnorm3 = norm_layer(embed_dims[3])
        # main  encoder
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3],
            use_mamba=use_mamba_stages[3],
            mamba_chunk_size=mamba_chunk_size,
            mamba_parallel=mamba_parallel)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        embed_dims=[64, 128, 320, 512]
        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        x2, H2, W2 = self.mini_patch_embed1(x1.permute(0,2,1).reshape(B,embed_dims[0],H1,W1))

        # DynamicCrossAttention
        # x1_cross = rearrange(x1, 'b (h w) c -> b c h w', h=H1, w=W1)
        x1 = self.cross_attn1(x1, x2)

        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        
        for i, blk in enumerate(self.patch_block1):
            x2 = blk(x2, H2, W2)
        x2 = self.pnorm1(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        x1 = x1.permute(0,2,1).reshape(B,embed_dims[1],H1,W1)+x2
        x2, H2, W2 = self.mini_patch_embed2(x1)

        x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

        # DynamicCrossAttention
        # x1_cross = rearrange(x1, 'b (h w) c -> b c h w', h=H1, w=W1)
        x1 = self.cross_attn2(x1, x2)

        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        for i, blk in enumerate(self.patch_block2):
            x2 = blk(x2, H2, W2)
        x2 = self.pnorm2(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        
        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        x1 = x1.permute(0,2,1).reshape(B,embed_dims[2],H1,W1)+x2
        x2, H2, W2 = self.mini_patch_embed3(x1)

        x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

        # DynamicCrossAttention
        # x1_cross = rearrange(x1, 'b (h w) c -> b c h w', h=H1, w=W1)
        x1 = self.cross_attn3(x1, x2)

        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        
        for i, blk in enumerate(self.patch_block3):
            x2 = blk(x2, H2, W2)
        x2 = self.pnorm3(x2)
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        x1 = x1.permute(0,2,1).reshape(B,embed_dims[3],H1,W1)+x2
        x2_3d = x2.flatten(2).permute(0, 2, 1)  # [B, C, H, W] -> [B, N, C]
        x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

        # DynamicCrossAttention
        x1 = self.cross_attn4(x1, x2_3d)

        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # pdb.set_trace()
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

############################################################

class Mamba(nn.Module):
    """PyTorch_Mamba"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.chunk_size = chunk_size

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False
        )

        self.A = nn.Parameter(torch.randn(1, 1, d_state))
        self.B = nn.Parameter(torch.randn(1, 1, d_state))
        self.C = nn.Parameter(torch.randn(1, 1, d_state))
        self.D = nn.Parameter(torch.ones(dim))

        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.out_proj = nn.Linear(self.d_inner, dim)

        self.norm = nn.LayerNorm(dim)

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.uniform_(self.A, -1.0, -0.1)

        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.02)

        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        if self.in_proj.bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.0)
        
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def selective_scan(self, x):
        """实现选择性扫描机制"""
        B, L, _ = x.shape
        d_inner = self.d_inner
        A = -torch.exp(self.A)  # [1, d_state, 1]
        B_param = self.B  # [1, 1, d_state]
        C = self.C  # [1, 1, d_state]

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(L):
            u_t = x[:, t, :].unsqueeze(2)  # [B, 1, d_inner]
            h = torch.exp(A) * h + u_t * B_param
            y_t = torch.sum(h * C, dim=2)  # [B, d_inner]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        return y
    
    def chunked_scan(self, x, H, W):
        """分块扫描实现"""
        B, L, C_inner = x.shape
        A = -torch.exp(self.A)  # [1, d_state, 1]
        B_param = self.B  # [1, 1, d_state]
        C = self.C  # [1, 1, d_state]

        chunk_size = self.dynamic_chunk_size(H, W)
        num_chunks = (L + chunk_size - 1) // chunk_size
        
        pad_len = num_chunks * chunk_size - L
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), value=0)

        x = x.view(B, num_chunks, chunk_size, C_inner)

        h = torch.zeros(B, num_chunks, C_inner, self.d_state, device=x.device)
        outputs = []
        
        for t in range(chunk_size):
            x_t = x[:, :, t, :]  # [B, num_chunks, C_inner]
            x_t = x_t.unsqueeze(-1)  # [B, num_chunks, C_inner, 1]
            
            h = torch.exp(A) * h + x_t * B_param
            
            y_t = torch.sum(h * C, dim=-1)  # [B, num_chunks, C_inner]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=2)  # [B, num_chunks, chunk_size, C_inner]
        y = y.view(B, num_chunks * chunk_size, C_inner)

        if pad_len > 0:
            y = y[:, :L, :]
        
        return y
    
    def dynamic_chunk_size(self, H, W):
        """根据分辨率动态计算块大小"""
        seq_len = H * W
        if seq_len <= 4096:  # 64x64
            return 64
        elif seq_len <= 16384:  # 128x128
            return 128
        else:  # 256x256或更大
            return 256

    def forward(self, x, H, W):
        """ 
        x: 输入序列 [B, N, C] 其中 N = H*W
        返回: 增强后的序列 [B, N, C]
        """
        B, N, C = x.shape
        residual = x
        
        x_norm = self.norm(x)
        
        proj = self.in_proj(x_norm)  # [B, N, 2*di]
        x, gate = torch.split(proj, self.d_inner, dim=-1)
        x = x * torch.sigmoid(gate)  # 门控机制
        
        x = x.permute(0, 2, 1)  # [B, di, N]
        x = self.conv1d(x)[:, :, :N]  # 因果卷积
        x = x.permute(0, 2, 1)  # [B, N, di]
        x = self.chunked_scan(x,H,W)

        x = self.out_proj(x)  # [B, N, C]
        x = x + self.D * residual
        
        return x


class SpatialMambaBlock(nn.Module):
    """空间感知Mamba模块"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, directions=4, chunk_size=64, parallel=True):
        super().__init__()
        self.dim = dim
        self.directions = directions
        self.parallel = parallel
        self.chunk_size = chunk_size
        self.spiral_cache = {}
  
        self.dir_predict = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, directions)
        )

        self.mamba_layers = nn.ModuleList([
            Mamba(dim, d_state, d_conv, expand,
                chunk_size=chunk_size
            ) for _ in range(directions)
        ])

        self.norm = nn.LayerNorm(dim)

        self.scan_methods = [
            self.raster_scan,
            self.reverse_raster_scan,
            self.vertical_scan,
            self.reverse_vertical_scan
        ]
    
    def raster_scan(self, x, H, W):
        """光栅扫描顺序 (左->右, 上->下)"""
        return x
    
    def reverse_raster_scan(self, x, H, W):
        """反向光栅扫描 (右->左, 下->上)"""
        x_2d = x.view(x.size(0), H, W, x.size(2))
        x_2d = torch.flip(x_2d, [1, 2])
        return x_2d.view(x.size(0), H * W, x.size(2))
    
    def vertical_scan(self, x, H, W):
        """垂直扫描顺序 (上->下, 左->右)"""
        x_2d = x.view(x.size(0), H, W, x.size(2))
        x_2d = x_2d.permute(0, 2, 1, 3)
        return x_2d.contiguous().view(x.size(0), H * W, x.size(2))  

    def reverse_vertical_scan(self, x, H, W):    
        """反向垂直扫描顺序 (下->左, 右->左)"""  
        x_2d = x.view(x.size(0), H, W, x.size(2))
        x_2d = torch.flip(x_2d, [1]) 
        x_2d = torch.flip(x_2d, [2]) 
        x_2d = x_2d.permute(0, 2, 1, 3)
        return x_2d.contiguous().view(x.size(0), H * W, x.size(2))
        

    def forward(self, x, H, W):
        """ 
        x: 输入序列 [B, N, C] 其中 N = H*W
        返回: 增强后的序列 [B, N, C]
        """
        B, N, C = x.shape

        x_reshaped = x.permute(0, 2, 1)  # [B, C, N]

        pooled = F.adaptive_avg_pool1d(x_reshaped, 1)  # [B, C, 1]
        pooled = pooled.view(B, C)  # [B, C]

        dir_weights = F.softmax(self.dir_predict(pooled), dim=1)  # [B, directions]
        
        if self.parallel:
            outputs = []
            for i in range(self.directions):
                scanned_x = self.scan_methods[i](x, H, W)
                outputs.append(self.mamba_layers[i](scanned_x, H, W))
                
            outputs = torch.stack(outputs, dim=0)
            weighted_out = torch.einsum('dbnc,bd->bnc', outputs, dir_weights)
        else:
            outputs = []
            for i in range(min(self.directions, len(self.scan_methods))):
                scanned_x = self.scan_methods[i](x, H, W)
                mamba_out = self.mamba_layers[i](scanned_x, H, W)
                outputs.append(mamba_out)
            
            if len(outputs) < self.directions:
                for i in range(len(outputs), self.directions):
                    outputs.append(outputs[-1])

            weighted_out = torch.zeros_like(x)
            for i, out in enumerate(outputs):
                weight = dir_weights[:, i].view(B, 1, 1)
                weighted_out += weight * out

        return self.norm(weighted_out)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DynamicCrossAttention(nn.Module):
    """动态交叉注意力模块 (用于特征融合)"""
    def __init__(self, dim, context_dim, k=5):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.k = k
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        
        self.threshold_net = nn.Sequential(
            nn.Linear(dim, dim//4),
            nn.GELU(),
            nn.Linear(dim//4, 1)
        )
        
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

    def forward(self, x, context):
        """
        x: 当前特征 [B, N, C]
        context: 交叉特征 [B, M, C]
        """
        B, N, C = x.shape
        M = context.shape[1]
        
        x_norm = self.norm(x)
        context_norm = self.norm_context(context)
        
        Q = self.to_q(x_norm)  # [B, N, C]
        K = self.to_k(context_norm)  # [B, M, C]
        V = self.to_v(context_norm)  # [B, M, C]
        
        attn = torch.einsum('bic,bjc->bij', Q, K) / math.sqrt(C)  # [B, N, M]
        
        thresholds = self.threshold_net(x_norm).squeeze(-1)  # [B, N]
        mask = attn > thresholds.unsqueeze(-1)  # [B, N, M]
        attn = attn * mask.float()
        
        topk_val, topk_idx = torch.topk(attn, self.k, dim=-1)  # [B, N, k]

        sparse_attn = torch.zeros_like(attn).scatter(-1, topk_idx, topk_val)
        sparse_attn = F.softmax(sparse_attn, dim=-1)

        out = torch.einsum('bij,bjc->bic', sparse_attn, V)
        out = self.proj(out)
        
        return out + x

class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = nn.Parameter(torch.randn(1,48,dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        task_q = self.task_query
        if B>1:
            
            task_q = task_q.unsqueeze(0).repeat(B,1,1,1)
            task_q = task_q.squeeze(1)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q,size= (v.shape[2],v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block_dec(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, 
                 use_mamba=False,
                 mamba_chunk_size=64, mamba_parallel=True):
        super().__init__()
        self.use_mamba = use_mamba
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mamba = mamba_block
        if use_mamba:
            # Mamba配置
            self.mamba = SpatialMambaBlock(dim, 
                chunk_size=mamba_chunk_size, 
                parallel=mamba_parallel)
            self.mlp = None
        else:
            # 原始MLP实现
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                           act_layer=act_layer, drop=drop)
            self.mamba = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        if self.mamba:
            x = x + self.drop_path(self.mamba(self.norm2(x), H, W))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DecoderTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size//16, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[3])

        # transformer decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block_dec(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[3])

        cur += depths[0]
        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    
    def forward_features(self, x):
        x=x[3]
        B = x.shape[0]
        outs = []
        
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

class Tenc(EncoderTransformer):
    def __init__(self, mamba_chunk_size=64, mamba_parallel=True, **kwargs):
        use_mamba_stages = [False, False, True, True ]                  
        super(Tenc, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 4], mlp_ratios=[2, 2, 2, 2],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[4, 2, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,
            use_mamba=True,
            use_mamba_stages=use_mamba_stages,
            mamba_chunk_size=mamba_chunk_size,
            mamba_parallel=mamba_parallel,
            **kwargs
        )

class Tdec(DecoderTransformer):
    def __init__(self, **kwargs):
        super(Tdec, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class convprojection(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection,self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential( ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()        

    def forward(self,x1,x2):

        res32x = self.convd32x(x2[0])

        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,-1,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)
            
        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0,-1,0,0)
            res32x = F.pad(res32x,p2d,"constant",0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,0,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)

        res16x = res32x + x1[3]
        res16x = self.convd16x(res16x) 

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x) 
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x

class convprojection_base(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection_base,self).__init__()

        # self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential( ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()        

    def adjust_size(self, x, target):
        """调整特征图尺寸大小"""
        H_x, W_x = x.size(2), x.size(3)
        H_t, W_t = target.size(2), target.size(3)
    
        if H_x < H_t:
            pad_h = H_t - H_x
            x = F.pad(x, (0, 0, 0, pad_h), "constant", 0)
        elif H_x > H_t:
            x = x[:, :, :H_t, :]
 
        if W_x < W_t:
            pad_w = W_t - W_x
            x = F.pad(x, (0, pad_w, 0, 0), "constant", 0)
        elif W_x > W_t:
            x = x[:, :, :, :W_t]
        
        return x

    def forward(self, x1):
        res16x = self.convd16x(x1[3])
        
        res16x = self.adjust_size(res16x, x1[2])
        res8x = self.dense_4(res16x) + x1[2]
        
        res8x = self.convd8x(res8x)
        res8x = self.adjust_size(res8x, x1[1])
        res4x = self.dense_3(res8x) + x1[1]
        
        res4x = self.convd4x(res4x)
        res4x = self.adjust_size(res4x, x1[0])
        res2x = self.dense_2(res4x) + x1[0]
        
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x



class WMF_NET(nn.Module):

    def __init__(self, path=None, **kwargs):
        super(WMF_NET, self).__init__()

        self.wavelet_preprocessor = WaveletPreprocessor(wave='sym2', enhance_factor=[1.3, 1.2, 1.1])

        self.Tenc = Tenc()

        self.wavelet_fusions = nn.ModuleList([
            WaveletFeatureFusion(in_channels=64, reduction_ratio=16),
            WaveletFeatureFusion(in_channels=128, reduction_ratio=16),
            WaveletFeatureFusion(in_channels=320, reduction_ratio=16),
            WaveletFeatureFusion(in_channels=512, reduction_ratio=16)
        ])
        
        self.convproj = convprojection_base()

        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()
        
        if path is not None:
            self.load(path)

    def forward(self, x):
        x_wav = self.wavelet_preprocessor(x)

        x1 = self.Tenc(x_wav)

        fused_features = []
        for i, feat in enumerate(x1):
            if feat.dim() == 3:
                _, _, C = feat.shape
                H, W = int(feat.shape[1]**0.5), int(feat.shape[1]**0.5)
                feat = feat.view(-1, H, W, C).permute(0, 3, 1, 2)
            
            fused = self.wavelet_fusions[i](feat)
            fused_features.append(fused)

        x = self.convproj(fused_features)

        clean = self.active(self.clean(x))

        return clean

    def load(self, path):
        """
        Load checkpoint.
        """
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        model_state_dict_keys = self.state_dict().keys()
        checkpoint_state_dict_noprefix = strip_prefix_if_present(checkpoint['state_dict'], "module.")
        self.load_state_dict(checkpoint_state_dict_noprefix, strict=False)
        del checkpoint
        torch.cuda.empty_cache()



