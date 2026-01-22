# TGRS 2025
import torch
from torch import nn


class AdaFocalLoss(nn.Module):
    def __init__(self, loss_fcn):
        """Adaptive parameter adjustment"""
        super(AdaFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, target):
        # pred = torch.sigmoid(pred)
        # 计算面积自适应权重
        area_weight = self._get_area_weight(target)  # [N,1,1,1]
        smooth = 1

        intersection = pred.sigmoid() * target
        iou = (intersection.sum() + smooth) / (pred.sigmoid().sum() + target.sum() - intersection.sum() + smooth)
        iou = torch.clamp(iou, min=1e-6, max=1 - 1e-6).detach()
        # BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # 自带sigmoid
        BCE_loss = self.loss_fcn(pred, target)
        target = target.type(torch.long)
        at = target * area_weight + (1 - target) * (1 - area_weight)
        pt = torch.exp(-BCE_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)
        F_loss = (1 - pt) ** (1 - iou + 1e-6) * BCE_loss

        F_loss = at * F_loss
        return F_loss.sum()

    def _get_area_weight(self, target):
        # 小目标增强权重
        area = target.sum(dim=(1, 2))  # [N,1]
        return torch.sigmoid(1 - area / (area.max() + 1)).view(-1, 1, 1, 1)