import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module): #层归一化，解决小参数模型训练不稳定问题
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        # 可学习的缩放参数（weight）和偏移参数（bias），初始值分别为1和0（保持原始分布）
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format  # 张量格式标识，默认channels_first
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, ) # 归一化的维度
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        #手动实现层归一化
        elif self.data_format == "channels_first":
            # 计算通道维度（dim=1）的均值u和方差s
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            #归一化
            x = (x - u) / torch.sqrt(s + self.eps)
            #通过可学习参数weight/bias调整分布（恢复特征表达能力）
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class NormDownsample(nn.Module):
    def __init__(self,in_ch,out_ch,scale=0.5,use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm=use_norm  # 归一化开关（灵活控制训练时是否启用，适配不同层需求）
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        # 下采样核心：3x3卷积（提取局部特征）+ 双线性下采样（缩小尺寸 图像尺寸减半）
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
    def forward(self, x):
        x = self.down(x)  #  下采样+通道转换：x（B,in_ch,H,W）→ 卷积→下采样 → x（B,out_ch,H/2,W/2）
        x = self.prelu(x) #  激活函数
        #归一化 稳定训练
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x

class NormUpsample(nn.Module):
    def __init__(self, in_ch,out_ch,scale=2,use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm=use_norm #归一化开关
        if self.use_norm:
            self.norm=LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        # 上采样核心：3x3卷积（通道转换）+ 双线性上采样（放大尺寸）
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        # 跳连接融合：1x1卷积（通道压缩）——拼接后通道数为2*out_ch，需压缩回out_ch
        self.up = nn.Conv2d(out_ch*2,out_ch,kernel_size=1,stride=1, padding=0, bias=False)
            
    def forward(self, x,y):
        # x：解码器前一层的特征（小尺寸、高通道，如B,out_ch,H/2,W/2）
        # y：编码器对应层的跳连接特征（大尺寸、低通道，如B,out_ch,H,W）
        # 上采样+通道转换
        x = self.up_scale(x)
        #跳连接融合
        x = torch.cat([x, y],dim=1) #拼接后是2倍
        x = self.up(x) #1x1卷积压缩通道
        x = self.prelu(x)
        #可选归一化
        if self.use_norm:
            return self.norm(x)
        else:
            return x
 

