import torch
import torch.nn as nn

__all__ = ['HCMFA']


class CN_Layer(nn.Module):
    """Cross-Nonlocal Layer, CNL"""
    """交叉非局部层，用于捕捉不同特征空间中的长距离依赖关系。"""

    def __init__(self, high_dim, low_dim, flag=0):
        super(CN_Layer, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        # 定义g卷积层，用于生成注意力机制中的g特征
        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        # 定义theta卷积层，用于生成注意力机制中的theta特征
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        # 根据flag的不同，定义phi卷积层和W序列
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(self.high_dim), )
        # 初始化批量归一化层的权重和偏置
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        # 通过g卷积层生成g特征并转换为矩阵形式
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        # 通过theta卷积层生成theta特征，通过phi卷积层生成phi特征，并转换为矩阵形
        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        # 注意phi_x需要转置以匹配矩阵乘法的维度要求
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        # 计算能量矩阵，即theta和phi特征的矩阵乘法结果
        energy = torch.matmul(theta_x, phi_x)
        # 归一化能量矩阵以获得注意力分数
        attention = energy / energy.size(-1)

        # 通过注意力分数和g特征进行矩阵乘法，得到加权的g特征
        y = torch.matmul(attention, g_x)
        # 将结果重新转换为张量形式，以匹配原始输入的维度
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        # 通过W序列对y进行变换，以提升特征维度并进行批量归一化
        W_y = self.W(y)
        # 将变换后的特征与原始高维特征相加，实现特征融合
        z = W_y + x_h

        return z


class PN_Layer(nn.Module):
    """Pixel Nonlocal Layer,PNL"""
    """像素非局部层，用于捕捉空间上的长距离依赖关系。"""

    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PN_Layer, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        # 定义g, theta, phi卷积层，用于生成注意力机制中的特征
        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        # 初始化批量归一化层的权重和偏置
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)

        # 通过g, theta, phi卷积层生成对应的特征并转换为矩阵形式
        g_x = self.g(x_l).reshape(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).reshape(B, self.low_dim, -1)

        # 计算能量矩阵，即theta和phi特征的矩阵乘法结果
        energy = torch.matmul(theta_x, phi_x)
        # 归一化能量矩阵以获得注意力分数
        attention = energy / energy.size(-1)

        # 通过注意力分数和g特征进行矩阵乘法，得到加权的g特征
        y = torch.matmul(attention, g_x)
        # 将结果重新转换为张量形式，以匹配原始输入的维度
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        # 通过W序列对y进行变换，以提升特征维度并进行批量归一化
        W_y = self.W(y)
        # 将变换后的特征与原始高维特征相加，实现特征融合
        z = W_y + x_h
        return z


class HCMFA(nn.Module):
    def __init__(self, dim):
        super(HCMFA, self).__init__()
        low_dim, high_dim = dim, dim
        self.CN_L = CN_Layer(high_dim, low_dim)  # 实例化交叉非局部层
        self.PN_L = PN_Layer(high_dim, low_dim)  # 实例化像素非局部层

    def forward(self, DATA):
        x, x0 = DATA
        # 通过交叉非局部层
        z = self.CN_L(x, x0)
        # 再通过像素非局部层
        z = self.PN_L(z, x0)
        return z