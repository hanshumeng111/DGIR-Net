import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import *

class GatedDynamicCAB(nn.Module):
    """
    Cross Attention Block with:
    - [IMPROVE-A] Content-adaptive gating on cross-branch injection
    - [IMPROVE-B] Degradation-aware dynamic temperature (per-sample, per-head)
    """
    def __init__(self, dim, num_heads, bias, k_min=0.5, k_max=2.0):
        super().__init__()
        self.num_heads = num_heads

        # 原来的 head-wise temperature（保留为基础温度）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # q from x, kv from y（保持原结构）
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # [IMPROVE-A] 门控：控制 y->x 的注入强度
        # [修改后] 严格遵循 PDF 建议的空间门控 [B, 1, H, W]
        self.gate = nn.Sequential(
        nn.Conv2d(dim * 2, 1, kernel_size=1, bias=True),
        nn.Sigmoid()
        )
        # [IMPROVE-B] 动态 temperature
        self.k_min = k_min
        self.k_max = k_max
        self.t_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_heads),
            nn.Sigmoid()
        )

    @staticmethod
    def _simple_degradation_stats(feat, dark_th=0.1, eps=1e-6):
        B = feat.size(0)
        s = feat.mean(dim=1, keepdim=True)  # (B,1,H,W)
        s = s.sigmoid()  # 将统计映射到(0,1)
        x = s.view(B, -1)

        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        dark_ratio = (x < dark_th).float().mean(dim=1, keepdim=True)

        return torch.cat([mean, var, dark_ratio], dim=1)  # (B,3)

    def forward(self, x, y):
        b, c, h, w = x.shape

        # q, k, v
        q = self.q_dwconv(self.q(x))  # (B,C,H,W)
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        # reshape to multi-head
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # normalize
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # [IMPROVE-B] 动态 temperature
        stats = self._simple_degradation_stats(x)  # (B,3)
        t_scale = self.t_mlp(stats)                # (B, num_heads) in (0,1)
        t_scale = self.k_min + (self.k_max - self.k_min) * t_scale  
        t_scale = t_scale.view(b, self.num_heads, 1, 1)              
        temperature = self.temperature.unsqueeze(0) * t_scale       

        # attention
        attn = (q @ k.transpose(-2, -1)) * temperature
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        # [IMPROVE-A] 门控注入：提取出门控图 g 并返回！
        g = self.gate(torch.cat([x, y], dim=1))  # (B,C,1,1)
        out = out * g

        # ===== 核心修改：返回特征的同时，把门控矩阵交出去 =====
        return out, g

# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x

from net.transformer_utils import LayerNorm  

class HV_DGDI(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = GatedDynamicCAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
        # ===== 核心修改：接住 ffn 返回的门控图 g =====
        attn_out, g = self.ffn(self.norm(x), self.norm(y))
        x = x + attn_out
        x = x + self.gdfn(self.norm(x))
        return x, g

class I_DGDI(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = GatedDynamicCAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
        # ===== 核心修改：接住 ffn 返回的门控图 g =====
        attn_out, g = self.ffn(self.norm(x), self.norm(y))
        x = x + attn_out
        x = x + self.gdfn(self.norm(x))
        return x, g
