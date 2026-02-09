import torch
import torch.nn as nn
import torch.fft


class LTFFAConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Lightweight TFFAConv
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        padding = k // 2 if p is None else p

        # -------------------------- 1. 小波分支 (轻量化：DW+PW) --------------------------
        # 原始：两路标准卷积 -> 改进：共享或独立的 Depthwise 卷积 + Pointwise 卷积
        # 这里为了保留"DoG"和"墨西哥帽"提取不同特征的逻辑，我们使用独立的 DW 卷积

        # 分支通道数
        c_wavelet = c2 // 2

        # DoG 模拟: DW (提取形状) -> PW (线性组合)
        self.dog_dw = nn.Conv2d(c1, c1, kernel_size=k, stride=1, padding=padding, groups=c1, bias=False)
        self.dog_pw = nn.Conv2d(c1, c_wavelet, kernel_size=1, bias=False)

        # 墨西哥帽模拟: DW -> PW
        self.mexican_dw = nn.Conv2d(c1, c1, kernel_size=k, stride=1, padding=padding, groups=c1, bias=False)
        self.mexican_pw = nn.Conv2d(c1, c_wavelet, kernel_size=1, bias=False)

        self.wavelet_norm = nn.BatchNorm2d(c2)

        # -------------------------- 2. 傅里叶分支 (轻量化：Bottleneck) --------------------------
        # 原始：全通道 FFT -> 改进：先降维，再 FFT
        # 减少频域处理的通道数，例如只处理 1/2 的通道，大幅降低 FLOPs
        self.fft_dim = max(1, c2 // 2)

        # 降维投影
        self.fourier_proj = nn.Conv2d(c1, self.fft_dim, kernel_size=1, bias=False)
        # 频域特征处理 (实部+虚部 -> 目标通道)
        self.fourier_process = nn.Conv2d(self.fft_dim * 2, c2, kernel_size=1, bias=False)
        self.fourier_norm = nn.BatchNorm2d(c2)

        # -------------------------- 3. 空间分支 (轻量化：保持) --------------------------
        # 1x1 卷积本身已经很轻量，保留用于特征对齐
        self.spatial_conv = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.spatial_norm = nn.BatchNorm2d(c2)

        # -------------------------- 4. 注意力融合 (轻量化：Squeeze) --------------------------
        # 原始：Conv(3*c2 -> c2) -> 改进：Conv(3*c2 -> c2/4) -> ReLU -> Conv(c2/4 -> c2)
        # 类似于 SE-Block 的瓶颈结构，减少融合层的参数
        c_hidden = max(16, c2 // 4)
        self.attention = nn.Sequential(
            nn.Conv2d(c2 * 3, c_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_hidden, c2, kernel_size=1, bias=True),  # 只有最后一层保留 bias
            nn.Sigmoid()
        )

        self.final_conv = nn.Conv2d(c2, c2, kernel_size=1, bias=False)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # -------------------------- 小波分支 --------------------------
        # Depthwise
        dog_feat = self.dog_dw(x)
        mex_feat = self.mexican_dw(x)
        # Pointwise
        dog_out = self.dog_pw(dog_feat)
        mexican_out = self.mexican_pw(mex_feat)

        wavelet_out = torch.cat([dog_out, mexican_out], dim=1)
        wavelet_out = self.wavelet_norm(wavelet_out)
        wavelet_out = self.act(wavelet_out)

        # -------------------------- 傅里叶分支 --------------------------
        # 1. 先降维，减少 FFT 计算量
        x_fft_in = self.fourier_proj(x)

        # 2. 傅里叶变换
        # 使用 float32 保证精度，避免半精度下 NaN
        dtype = x.dtype
        fft = torch.fft.fft2(x_fft_in.float())
        fft_real = fft.real.to(dtype)
        fft_imag = fft.imag.to(dtype)

        # 3. 频域拼接与映射
        fourier_feat = torch.cat([fft_real, fft_imag], dim=1)
        fourier_out = self.fourier_process(fourier_feat)
        fourier_out = self.fourier_norm(fourier_out)
        fourier_out = self.act(fourier_out)

        # -------------------------- 空间分支 --------------------------
        spatial_out = self.spatial_conv(x)
        spatial_out = self.spatial_norm(spatial_out)
        spatial_out = self.act(spatial_out)

        # -------------------------- 注意力融合 --------------------------
        concat_feat = torch.cat([wavelet_out, fourier_out, spatial_out], dim=1)

        # 计算注意力权重 (Bottleneck 结构)
        attention_weights = self.attention(concat_feat)

        # 加权融合
        fused = (wavelet_out + fourier_out + spatial_out) * attention_weights

        # 也可以保留原始的逐项加权逻辑：
        # fused = wavelet_out * attention_weights + fourier_out * attention_weights + ...
        # 但既然维度一致，上面的写法利用广播机制在内存上可能略快一点，效果是一样的

        fused = self.final_conv(fused)

        return fused


# 测试代码对比
if __name__ == "__main__":
    from thop import profile  # 需要安装 thop: pip install thop
    import copy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    c_in, c_out = 64, 64
    input_tensor = torch.randn(1, c_in, 64, 64).to(device)  # Batch=1 for thop

    # 初始化模型
    model_light = LTFFAConv(c_in, c_out, k=3).to(device)

    # 打印测试
    output = model_light(input_tensor)
    print(f"LTFFAConv Input: {input_tensor.shape}")
    print(f"LTFFAConv Output: {output.shape}")

    # 简单的 FLOPs/Params 估算
    try:
        flops, params = profile(model_light, inputs=(input_tensor,), verbose=False)
        print(f"Lightweight FLOPs: {flops / 1e6:.2f}M, Params: {params / 1e3:.2f}K")
    except Exception as e:
        print("Install 'thop' library to see FLOPs count.")