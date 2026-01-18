import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CEM']


class CEM(nn.Module):
    def __init__(self, channel):
        super(CEM, self).__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x1_amp1 = torch.mean(x, dim=1, keepdim=True)
        x1_amp1 = x - x1_amp1
        att = self.adaptive_avg_pool(x1_amp1)
        att = F.softmax(att, 1)
        return att * x


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    CEM_module = CEM(32)
    # 创建一个随机输入张量，假设批量大小为1，通道数为32，图像尺寸为64x64
    input = torch.randn(1, 32, 64, 64)

    output = CEM_module(input)
    # 输出结果的形状
    print("输入张量的形状：", input.shape)
    print("输出张量的形状：", output.shape)