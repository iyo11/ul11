import torch
import torch.nn as nn
import torch.fft


class TFFAConv(nn.Module):
    # 修改 1: 标准化参数名 (c1, c2, k, s) 以匹配 YOLO 的 parse_model
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # 为了保证特征图大小不变 (Stride=1)，padding 设为 k // 2
        # 如果 YOLO 传入了 s (stride) 但本模块设计为不改变尺寸，我们忽略输入的 s 或强制为 1
        # 这里保留原始逻辑，padding=k//2 确保 H,W 不变
        padding = k // 2 if p is None else p

        # 1. 小波分支：DoG和墨西哥帽小波
        # 注意：c2//2 可能需要确保 c2 是偶数，YOLO 通道扩充通常是倍数，问题不大
        self.dog_conv = nn.Conv2d(c1, c2 // 2, kernel_size=k, padding=padding, stride=1)
        self.mexican_conv = nn.Conv2d(c1, c2 // 2, kernel_size=k, padding=padding, stride=1)
        self.wavelet_norm = nn.BatchNorm2d(c2)

        # 2. 傅里叶分支：频域特征提取
        self.fourier_conv = nn.Conv2d(c1 * 2, c2, kernel_size=1)  # 实部+虚部
        self.fourier_norm = nn.BatchNorm2d(c2)

        # 3. 空间分支：逐点卷积
        self.spatial_conv = nn.Conv2d(c1, c2, kernel_size=1)
        self.spatial_norm = nn.BatchNorm2d(c2)

        # 注意力融合门控
        self.attention = nn.Sequential(
            nn.Conv2d(c2 * 3, c2, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(c2, c2, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        # -------------------------- 小波分支 --------------------------
        dog_out = self.dog_conv(x)
        mexican_out = self.mexican_conv(x)
        wavelet_out = torch.cat([dog_out, mexican_out], dim=1)
        wavelet_out = self.wavelet_norm(wavelet_out)
        wavelet_out = self.act(wavelet_out)

        # -------------------------- 傅里叶分支 --------------------------
        # 傅里叶变换（实部+虚部）
        # ⚠️ 注意：如果输入是半精度(FP16)，fft2可能会报错，建议转为float32计算后再转回
        dtype = x.dtype
        fft = torch.fft.fft2(x.float())
        fft_real = fft.real.to(dtype)
        fft_imag = fft.imag.to(dtype)

        fourier_feat = torch.cat([fft_real, fft_imag], dim=1)
        fourier_out = self.fourier_conv(fourier_feat)
        fourier_out = self.fourier_norm(fourier_out)
        fourier_out = self.act(fourier_out)

        # -------------------------- 空间分支 --------------------------
        spatial_out = self.spatial_conv(x)
        spatial_out = self.spatial_norm(spatial_out)
        spatial_out = self.act(spatial_out)

        # -------------------------- 注意力融合 --------------------------
        concat_feat = torch.cat([wavelet_out, fourier_out, spatial_out], dim=1)
        attention_weights = self.attention(concat_feat)

        fused = wavelet_out * attention_weights + fourier_out * attention_weights + spatial_out * attention_weights
        fused = self.final_conv(fused)

        return fused


if __name__ == "__main__":
    from thop import profile  # 需要安装 thop: pip install thop
    import copy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    c_in, c_out = 64, 64
    input_tensor = torch.randn(1, c_in, 64, 64).to(device)  # Batch=1 for thop

    # 初始化模型
    model_light = TFFAConv(c_in, c_out, k=3).to(device)

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