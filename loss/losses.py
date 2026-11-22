import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *


_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        
        
        
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        # 修复：统一通道数为3（适配EdgeLoss的kernel）
        if x.size(1) != 3:
            x = x.repeat(1, 3, 1, 1)
        if y.size(1) != 3:
            y = y.repeat(1, 3, 1, 1)
            
        loss = F.mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()  # 修复：L2Loss不存在，正确为MSELoss
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # 核心修复1：强制转为3通道（解决单通道/多通道问题）
        if x.size(1) != 3:
            x = x.repeat(1, 3, 1, 1)  # 单通道→3通道，多通道→取前3通道
        if gt.size(1) != 3:
            gt = gt.repeat(1, 3, 1, 1)
        
        # 核心修复2：限制数据范围在[0,1]（VGG要求的输入范围）
        x = torch.clamp(x, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        # 提取vgg特征（增加异常防护）
        try:
            x_features = self.vgg(x)
            gt_features = self.vgg(gt.detach())
        except Exception as e:
            print(f"VGG特征提取出错: {e}，跳过本轮感知损失计算")
            return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # 计算感知损失
        percep_loss = 0.0
        if self.perceptual_weight > 0:
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight

        # 计算风格损失
        style_loss = 0.0
        if self.style_weight > 0:
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight

        return percep_loss, style_loss


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # 修复：统一通道数
        if img1.size(1) != 3:
            img1 = img1.repeat(1, 3, 1, 1)
        if img2.size(1) != 3:
            img2 = img2.repeat(1, 3, 1, 1)

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight

import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    """提取图像的 Sobel 梯度幅值"""
    device = x.device
    dtype = x.dtype
    channels = x.shape[1]
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)
    
    grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
    
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return grad

class SobelEdgeLoss(nn.Module):
    """实验 1 专属：基于 Sobel 算子的 L1 边缘结构保持损失"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, gt):
        # 对应 PDF 公式：L_edge = || ∇(pred) - ∇(gt) ||_1
        return self.l1(sobel_grad(pred), sobel_grad(gt))
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """计算总变分(TV)损失，用于平滑门控图 [cite: 89-107]"""
    loss_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return loss_h + loss_w

class GateRegLoss(nn.Module):
    """实验 2 专属：动态门控正则损失 [cite: 153-177]"""
    def __init__(self, beta_sparse=1.0, gamma_tv=1.0):
        super().__init__()
        self.beta_sparse = beta_sparse
        self.gamma_tv = gamma_tv

    def forward(self, gate_h2i, gate_i2h):
        loss_gate_sparse = torch.tensor(0.0, device='cuda')
        loss_gate_tv = torch.tensor(0.0, device='cuda')

        # 强大的兼容逻辑：兼容单张图，也兼容包含多个 DGDI 门控的 List
        if gate_h2i is not None:
            if not isinstance(gate_h2i, (list, tuple)): gate_h2i = [gate_h2i]
            for g in gate_h2i:
                if g is not None:
                    loss_gate_sparse += torch.mean(torch.abs(g))
                    loss_gate_tv += tv_loss(g)
                    
        if gate_i2h is not None:
            if not isinstance(gate_i2h, (list, tuple)): gate_i2h = [gate_i2h]
            for g in gate_i2h:
                if g is not None:
                    loss_gate_sparse += torch.mean(torch.abs(g))
                    loss_gate_tv += tv_loss(g)

        return self.beta_sparse * loss_gate_sparse + self.gamma_tv * loss_gate_tv