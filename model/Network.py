
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import os
import cv2
import torch.fft as fft
import random


def fft_conv(x, psf, flip_psfs=False):
    """
    完全向量化的、基于FFT的分组卷积/相关函数。
    Args:
        x (torch.Tensor): 输入张量, shape: [B, C, H_in, W_in]
        psf (torch.Tensor): 卷积核张量, shape: [C, 1, H_k, W_k]
    Returns:
        torch.Tensor: 计算结果, shape: [B, C, H_in, W_in]
    """
    B, C, H_in, W_in = x.shape
    _, _, H_k, W_k = psf.shape
    # 1. 确定FFT尺寸
    fft_h = H_in + H_k - 1
    fft_w = W_in + W_k - 1
    fft_size = (fft_h, fft_w)
    # 2. 准备卷积核
    kernel = psf
    if flip_psfs:
        kernel = torch.flip(psf, [2, 3])
    # 3. 计算FFT (在所有Batch和Channel上并行执行)
    x_fft = torch.fft.rfft2(x, s=fft_size)
    kernel_fft = torch.fft.rfft2(kernel, s=fft_size)
    # 4. 频域相乘 (利用广播机制)
    # x_fft: [B, C, ...], kernel_fft: [C, 1, ...] -> PyTorch自动广播
    # 如果kernel_fft的batch维是C，需要对齐 [1, C, ...]
    result_fft = x_fft * kernel_fft.squeeze(0).unsqueeze(0)  # 确保广播正确
    # 5. IFFT (在所有Batch和Channel上并行执行)
    full_conv = torch.fft.irfft2(result_fft, s=fft_size)
    # 6. 裁剪 (一次性在所有维度上操作)
    full_conv = full_conv[..., :H_in + H_k - 1, :W_in + W_k - 1]
    crop_start_h = (full_conv.shape[-2] - H_in) // 2
    crop_start_w = (full_conv.shape[-1] - W_in) // 2
    result = full_conv[...,
             crop_start_h: crop_start_h + H_in,
             crop_start_w: crop_start_w + W_in]
    result = result.squeeze(0)
    return result

def gradient_descent_step(x_list, y_sum, psfs, step_size):
    """
    Args:
        x_list (torch.Tensor): [10, 1, 800, 800], 待更新的图像列表
        y_sum (torch.Tensor): [1, 800, 800], 观测值
        psfs (list of torch.Tensor): 10个 [1, 1, 800, 800] 的卷积核列表
        step_size (float): 更新步长 (rho)
    """
    psfs_stacked = torch.stack(psfs).squeeze(1)
    Ax_list = fft_conv(x_list, psfs_stacked, flip_psfs=False)
    Ax_sum = torch.sum(Ax_list, dim=0, keepdim=True)  # [1, 1, 800, 800]
    b, c, n, m = x_list.shape
    ln1 = nn.LayerNorm([c, n, m]).cuda()
    residual = ln1(Ax_sum) - ln1(y_sum)  # Compute the residual
    residual_repeated = residual.repeat(10, 1, 1, 1)  # [1, 10, 800, 800]
    gradient = fft_conv(residual_repeated, psfs_stacked, flip_psfs=True)
    updated_x = x_list + torch.div(gradient, step_size+1)
    return updated_x

def initialize_X(Y_total, A_list, epsilon=1e-6, power_weighting=True):
    """
    从叠加测量值Y_total初始化各X的估计值
    参数：
        Y_total : 叠加后的测量值 [1, 1, H, W]
        A_list : PSF列表 [10, 1, H, W]
        epsilon : 正则化参数
        power_weighting : 是否使用功率加权分配
    返回：
        X_init_list : 初始化后的X估计列表 [10, 1, H, W]
    """
    device = Y_total.device
    n_psf = len(A_list)
    H, W = Y_total.shape[-2:]
    # 将Y_total转换到频域
    Y_fft = fft.fft2(Y_total.squeeze())  # [H, W] complex
    # 存储各PSF的频域响应和能量
    A_fft_list = []
    energy_list = []
    # 预处理所有PSF
    for A in A_list:
        # PSF中心化
        A_centered = torch.roll(A.squeeze(), shifts=(A.shape[-2] // 2, A.shape[-1] // 2), dims=(-2, -1))
        # 频域变换
        A_fft = fft.fft2(A_centered)
        A_fft_list.append(A_fft)
        energy_list.append(torch.abs(A_fft).pow(2).mean())   ##   energy_list.append(torch.abs(A_fft).pow(2).sum())
    # 计算加权系数
    weights = torch.stack(energy_list) if power_weighting else torch.ones(n_psf, device=device)
    weights = weights / weights.sum()
    X_init_list = []
    for i in range(n_psf):
        # 获取当前PSF的频域响应
        A_fft = A_fft_list[i]
        # 计算逆滤波器 (维纳滤波形式)
        H_conj = torch.conj(A_fft)
        H_energy = torch.abs(A_fft).pow(2)
        inverse_filter = H_conj / (H_energy + epsilon * H_energy.max())
        # 应用加权系数
        weighted_filter = inverse_filter * weights[i]
        # 频域反卷积
        X_fft = Y_fft * weighted_filter
        # 时域重建
        X_est = fft.ifft2(X_fft).abs()
        # 后处理
        X_est = X_est / (X_est.max())  # 归一化 X_est = X_est / (X_est.max() + 1e-8)
        X_est = X_est # [1, 1, H, W]
        X_init_list.append(X_est)
    return torch.stack(X_init_list).unsqueeze(1)

def correlate(image, kernel):
    """
    Performs 2D correlation (effectively the "transpose" of convolution).
    Args:
        image (torch.Tensor): Input image.
        kernel (torch.Tensor): Convolution kernel.
    Returns:
        torch.Tensor: Correlated image.
    """
    # Flip the kernel for correlation
    flipped_kernel = torch.flip(kernel, [2, 3])
    out = F.conv2d(image, flipped_kernel, padding='same')
    return out

def MultilayerSummaration(x_list, psf_list):
    """
    Forward model for snapshot measurement
    Args:
        x_list (torch.Tensor): Input image.
        psf_list (torch.Tensor): Convolution kernel.
    Returns:
        torch.Tensor: snapshot measurement .
    """
    psf = torch.stack(psf_list).squeeze(1)
    meas0 = fft_conv(x_list, psf)
    meas = torch.sum(meas0, 0)
    meas = meas.unsqueeze(1)
    return meas

class DepthAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, (dim // 2) * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        B_, N, C = x.shape
        C = C // 2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class CTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.DSAB = DepthAttention(dim, num_heads=4)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 1)
        )
    def forward(self, x):
        b, c, h, w = x.shape
        scb_out = self.scb(x)
        tsab_in = einops.rearrange(x, "b c h w->  (h w) b c")
        tsab_out = self.tsab(tsab_in)
        tsab_out = einops.rearrange(tsab_out, " (h w) b c->b c h w", h=h, w=w)
        ffn_in = scb_out + tsab_out + x
        ffn_out = self.ffn(ffn_in) + ffn_in
        return ffn_out

class ResDNetBlock(nn.Module):
    def __init__(self, dim, group_num):
        super().__init__()
        self.CTB_list = nn.ModuleList()
        self.group_num = group_num
        group_dim = dim // group_num
        self.dense_conv = nn.ModuleList()
        for i in range(group_num):
            self.CTB_list.append(CTBlock(group_dim))
            if i > 0:
                self.dense_conv.append(
                    nn.Sequential(
                        nn.Conv2d(group_dim * (i + 1), group_dim, 1),
                        nn.LeakyReLU(inplace=True)
                    )
                )
        self.last_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        input_list = torch.chunk(x, chunks=self.group_num, dim=1)
        cf_in = input_list[0]
        out_list = []
        cf_out = self.CTB_list[0](cf_in)
        out_list.append(cf_out)
        for i in range(1, self.group_num):
            in_list = out_list.copy()
            in_list.append(input_list[i])
            cf_in = torch.cat(in_list, dim=1)
            cf_in = self.dense_conv[i - 1](cf_in)
            cf_out = self.cformer_list[i](cf_in)
            out_list.append(cf_out)
        out = torch.cat(out_list, dim=1)
        out = self.last_conv(out)
        out = x + out
        return out


class ResidualGroup(nn.Module):
    def __init__(self, dim=128, group_num=4):
        super().__init__()
        self.residual_list = nn.Sequential(*[ResDNetBlock(dim=dim, group_num=group_num) for k in range (1)])
    def forward(self, x):
        x = self.residual_list(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, color_channels=1, width=64, middle_blk_num=6, enc_blk_nums=[1, 2, 4], dec_blk_nums=[1, 1, 1],
                 first_stage=False, final_stage=False):
        super(BasicBlock, self).__init__()
        self.first_stage = first_stage
        self.final_stage = final_stage
        self.embedding = nn.Conv2d(in_channels=color_channels, out_channels=width, kernel_size=3, padding=1, stride=1,
                                   groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        if not first_stage:
            self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.decoders = nn.ModuleList()
        if self.final_stage:
            self.mapping = nn.Conv2d(in_channels=width, out_channels=1, kernel_size=3, padding=1, stride=1,
                                     groups=1, bias=True)
        else:
            self.mapping = nn.Conv2d(in_channels=width, out_channels=color_channels, kernel_size=3, padding=1, stride=1,
                                 groups=1, bias=True)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            if not first_stage:
                self.convs1.append(
                    # SkipConnectionAttention(dim=chan*2, dim_out=chan)
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
                )
            self.encoders.append(
                nn.Sequential(
                    *[ResidualGroup(dim=chan, group_num=4) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[ResidualGroup(dim=chan, group_num=4) for _ in range(middle_blk_num)]
        )
        if not first_stage:
            self.middle_blks = nn.Sequential(
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False),
                *[ResidualGroup(dim=chan, group_num=4) for _ in range(middle_blk_num)])
        else:
            self.middle_blks = nn.Sequential(
                *[ResidualGroup(dim=chan, group_num=4) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.convs2.append(
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
            )
            self.decoders.append(
                nn.Sequential(
                    *[ResidualGroup(dim=chan, group_num=4) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, last_decs=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.embedding(inp)
        encs = []
        decs = []
        if not self.first_stage:
            for encoder, down, conv1, last_dec in zip(self.encoders, self.downs, self.convs1, last_decs[::-1]):
                x = conv1(torch.cat([x, last_dec], dim=1))
                x = encoder(x)
                encs.append(x)
                x = down(x)
        else:
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                encs.append(x)
                x = down(x)
        if not self.first_stage:
            x = self.middle_blks(torch.cat([x, last_decs[::-1][2]], dim=1))
        else:
            x = self.middle_blks(x)
        decs.append(x)
        for decoder, up, conv2, enc_skip in zip(self.decoders, self.ups, self.convs2, encs[::-1]):
            x = up(x)
            x = conv2(torch.cat([x, enc_skip], dim=1))
            x = decoder(x)
            decs.append(x)
        x = self.mapping(x)
        # x = x + inp
        return x[:, :, :H, :W], decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def manual_rotate_psf(psf_data, angle_degrees, interpolation='linear'):
    """
    手动实现PSF旋转，提供更精确的控制
    """
    angle_rad = np.radians(angle_degrees)
    h, w = psf_data.shape
    # 创建输出数组
    rotated = np.zeros_like(psf_data)
    # 中心坐标
    center_y, center_x = h // 2, w // 2
    # 创建旋转后的坐标网格
    y_indices, x_indices = np.indices((h, w))
    y_centered = y_indices - center_y
    x_centered = x_indices - center_x
    # 应用反向旋转变换
    cos_angle = np.cos(-angle_rad)
    sin_angle = np.sin(-angle_rad)
    x_original = x_centered * cos_angle - y_centered * sin_angle + center_x
    y_original = x_centered * sin_angle + y_centered * cos_angle + center_y
    # 确保坐标在有效范围内
    valid_mask = (x_original >= 0) & (x_original < w - 1) & \
                 (y_original >= 0) & (y_original < h - 1)
    if interpolation == 'nearest':
        # 最近邻插值
        x_original_int = np.round(x_original).astype(int)
        y_original_int = np.round(y_original).astype(int)
        rotated[valid_mask] = psf_data[y_original_int[valid_mask],
                                       x_original_int[valid_mask]]
    else:
        # 双线性插值
        x_floor = np.floor(x_original).astype(int)
        y_floor = np.floor(y_original).astype(int)
        x_ceil = x_floor + 1
        y_ceil = y_floor + 1
        # 权重计算
        x_weight = x_original - x_floor
        y_weight = y_original - y_floor
        # 确保不越界
        x_ceil = np.clip(x_ceil, 0, w - 1)
        y_ceil = np.clip(y_ceil, 0, h - 1)
        x_floor = np.clip(x_floor, 0, w - 1)
        y_floor = np.clip(y_floor, 0, h - 1)
        # 双线性插值
        top_left = psf_data[y_floor, x_floor] * (1 - x_weight) * (1 - y_weight)
        top_right = psf_data[y_floor, x_ceil] * x_weight * (1 - y_weight)
        bottom_left = psf_data[y_ceil, x_floor] * (1 - x_weight) * y_weight
        bottom_right = psf_data[y_ceil, x_ceil] * x_weight * y_weight
        rotated = top_left + top_right + bottom_left + bottom_right
    return rotated

def load_psf():
    """Load pre-calibrated DH-PSFs"""
    img_names = os.listdir('/path/to/Calibrated_PSFs/')
    num = len(img_names)
    img_names.sort()
    depth_list = []
    psf_list = []
    for i in range(num):
        image = cv2.imread(os.path.join('/path/to/Calibrated_PSFs', img_names[i]), cv2.IMREAD_GRAYSCALE)
        print('img_name:', img_names[i])
        temp1 = image.copy()
        ## pre-processing
        temp1[1000:1100, 1200:1300] = 0
        psf = image - temp1
        psf = psf[645:1445, 875:1675]  ## 800
        psf = torch.from_numpy(psf).cuda().float()
        psf = psf.unsqueeze(0).unsqueeze(1)
        psf_list.append(psf)

    return num, psf_list, depth_list

def normlization0_1(meas):
    """normlization of input"""
    meas = meas.squeeze(0).squeeze(1)
    meas_min = meas.min()
    meas_range = meas.max() - meas_min
    # 处理标准差为0的情况
    if meas_range == 0:
        X = torch.zeros_like(meas, dtype=torch.float32)
    else:
        X = (meas - meas_min) / meas_range
    return X


def add_gaussian_noise(tensor, mean=0.0, std=0.1):
    """给张量添加高斯噪声"""
    # 生成与输入张量形状相同的高斯噪声
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

def add_poisson_noise(tensor, lam=1.0):
    """给张量添加泊松噪声"""
    # 泊松噪声要求输入为非负值，先保存原始范围
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    # 归一化到 [0, lam] 范围
    normalized = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8) * lam
    # 生成泊松噪声并添加
    noise = torch.poisson(normalized)
    noisy_tensor = normalized + noise
    # 恢复到原始范围
    result = noisy_tensor / (lam + 1e-8) * (tensor_max - tensor_min) + tensor_min
    return result

class PGDUN(nn.Module):
    def __init__(self, imag_size, meas_size, img_channels=1, channels=64, mid_blocks=6, enc_blocks=[1, 2, 4],
                 dec_blocks=[1, 1, 1], stages=7, matrix_train=True, only_test=False):
        super().__init__()
        self.stages = stages
        self.imag_size = imag_size
        self.only_test = only_test
        self.psf_list = load_psf()
        self.ln1 = nn.LayerNorm([img_channels, 1, imag_size[0], imag_size[1]])
        self.mu = nn.Parameter(torch.Tensor([0.0001]).repeat(stages))
        self.denoisers = nn.ModuleList([])
        self.denoisers.append(BasicBlock(color_channels=img_channels, width=channels, middle_blk_num=mid_blocks,
                                         enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks, first_stage=True))
        for i in range(stages - 2):
            self.denoisers.append(BasicBlock(color_channels=img_channels, width=channels, middle_blk_num=mid_blocks,
                                             enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks))
        self.denoisers.append(BasicBlock(color_channels=img_channels, width=channels, middle_blk_num=mid_blocks,
                                         enc_blk_nums=enc_blocks, dec_blk_nums=dec_blocks, final_stage=True))
    def prox_g(self, GT, Z, a, b=1):
        X = b * GT + a * Z
        X = X / (a + b)
        return X

    def forward(self, X):
        X = X.permute(3, 0, 1, 2)
        X_gt = torch.sum(X, dim=0).unsqueeze(0)
        n, psf, dep = self.psf_list
        meas = MultilayerSummaration(X, psf)
        meas = normlization0_1(meas)
        # meas = add_gaussian_noise(meas, 0, 0.1)
        # meas = add_poisson_noise(meas, 50)

        # Testing for real data!
        # Simply replace the meas with real data in 'real data' Folder
        # Image = cv2.imread('/dataset/test——image.png', cv2.IMREAD_GRAYSCALE)
        # image_array = np.array(Image) * 1
        # meas = torch.from_numpy(image_array).float().cuda()
        # meas = meas[0:800, 500:1300]  ##1600
        # meas = normlization0_1(meas)
        # meas = meas.unsqueeze(0).unsqueeze(1)

        ## Initialization & Reconstruction
        X0 = initialize_X(meas, psf)
        print('X0:', X0.shape)
        X_out=[]
        for i in range(self.stages):
            ## For DUN model
            mu = self.mu[i]
            Z = gradient_descent_step(X0, meas, psf, mu)
            ## For E2E mode
            # Z = X0
            Z = self.ln1(Z)
            Z = Z.permute(1, 0, 2, 3)
            if i == 0:
                X0, features = self.denoisers[i](Z)
                X_out.append(X0)
            elif i==(self.stages-1):
                X1, features = self.denoisers[i](Z, features)
            else:
                X0, features = self.denoisers[i](Z, features)
                X_out.append(X0)
            X0 = X0.permute(1, 0, 2, 3)
        return X, X_out, X1, X_gt
