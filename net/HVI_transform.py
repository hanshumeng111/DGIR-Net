import torch
import torch.nn as nn

pi = 3.141592653589793

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # 定义密度参数k，用于消除黑色噪声
        #开关变量 控制饱和度和亮度调节是否生效
        self.gated = False
        self.gated2= False
        self.alpha = 1.0 #调节图像亮度
        self.alpha_s = 1.3 #调节图像饱和度
        self.this_k = 0  #存储当前k值，用于后期逆转换
        
    def HVIT(self, img): #RGB到HSV到HVI 关键是极化HS平面以及可学习强度函数Ck
        eps = 1e-8 #定义极小的常量 防止梯度爆炸
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes) #创建一个专门存储色调（Hue） 的张量，并将其移动到与img一致的设备和数据类型
        value = img.max(1)[0].to(dtypes) #计算亮度图V I_max(x) = max(R,G,B)
        img_min = img.min(1)[0].to(dtypes) #每个像素RGB的最小值，用于饱和度和色调计算
        #色调HUE的计算 （4种不同情况）
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6
        hue[img.min(1)[0]==value] = 0.0 #所有通道相等时，hue=0
        hue = hue/6.0 #归一化到【0，1】

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0
        #维度扩充为(B,1,H,W) 在维度1处新增维度
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        # ====================== 开始修改（共4行，替换原2行错误代码）======================
        # 原错误代码：
        # k = self.dynamic_k #获取当前训练的k值
        # self.this_k = k.item() #缓存
        # 新正确代码：
        k = self.density_k #获取当前训练的k值
        self.this_k = k.item() #缓存
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k) #色彩敏感因子 增强低光区域的色彩信息 实现强度压缩函数Ck
        #极化HS平面
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        #计算HVI中的色彩分量
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        #强度分量沿用HSV中的value
        I = value
        xyz = torch.cat([H, V, I],dim=1) #拼接 H、V、I 得到(B,3,H,W)的 HVI 张量，完成正向转换
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:] #拆分HVI的三个通道 （水平色彩分量H、垂直色彩分量V、强度 I）
        
        # 避免数值溢出
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
         
        #还原HSV空间的色调和饱和度
        v = I #HVI的强度分量直接作为HSV的亮度V
        k = self.this_k #调用HVIT中的k值，保持转换一致性
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k) #复用正向转换的Ck函数
        #还原HS分量
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V + eps,H + eps) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        if self.gated:
            s = s * self.alpha_s #通过self.gated开关控制是否乘以α_S（self.alpha_s）
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        #从HSV空间转换到RGB空间 定义r,g,b张量存储输出通道，按 HSV→RGB 的标准分段公式计算
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        #将Hue分成6个颜色段，通过插值计算RGB分量
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi # 插值系数（0~1），用于平滑过渡颜色
        p = v * (1. - s) # 辅助变量：亮度×(1-饱和度)
        q = v * (1. - (f * s))  # 辅助变量：亮度×(1-插值系数×饱和度)
        t = v * (1. - ((1. - f) * s))  # 辅助变量：亮度×(1-(1-插值系数)×饱和度)
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        #维度扩充加拼接  从(B,H,W)→(B,1,H,W)，再拼接成(B,3,H,W)的sRGB   4维张量    
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
    

# import torch
# import torch.nn as nn
# import math

# pi = 3.141592653589793

# class DynamicK(nn.Module):
#     def __init__(self, k_base=0.2, k_range=0.1):
#         super(DynamicK, self).__init__()
#         self.k_base, self.k_range = k_base, k_range
#         self.mlp = nn.Sequential(
#             nn.Linear(6, 32), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(32, 16), nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(16, 1), nn.Tanh()
#         )

#     def forward(self, I): 
#         B = I.shape[0]
#         x = I.view(B, -1)
        
#         # 1. 全局特征，强制 (B, 1)
#         mean = x.mean(dim=1).view(B, 1)
#         var = x.var(dim=1).view(B, 1)
#         dark_p = (x < 0.1).float().mean(dim=1).view(B, 1)

#         feats = [mean, var, dark_p]
        
#         # 2. 局部特征
#         for ks in [5, 11, 21]:
#             lm = nn.functional.avg_pool2d(I, ks, 1, ks//2)
#             lv = torch.clamp(nn.functional.avg_pool2d(I**2, ks, 1, ks//2) - lm**2, min=1e-8)
#             # 【关键修复】强制 view(B, 1)
#             lc = torch.sqrt(lv).view(B, -1).mean(dim=1).view(B, 1)
#             feats.append(lc)
            
#         # 现在所有特征都是 (B, 1)，cat 不会报错
#         return self.k_base + self.k_range * self.mlp(torch.cat(feats, 1))

# class RGB_HVI(nn.Module):
#     def __init__(self):
#         super(RGB_HVI, self).__init__()
#         self.dynamic_k_gen = DynamicK()
#         self.this_k = None 

#     def HVIT(self, img):
#         eps = 1e-6
#         V = img.max(1)[0].unsqueeze(1)
#         m = img.min(1)[0].unsqueeze(1)
#         curr_k = self.dynamic_k_gen(V)
#         self.this_k = curr_k.detach()
        
#         base = torch.clamp(torch.sin(V * 0.5 * pi), min=1e-4, max=1.0)
#         color_sen = base.pow(curr_k.view(-1, 1, 1, 1))
        
#         diff = V - m + eps
#         r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
#         h = torch.zeros_like(V)
#         mask_r, mask_g, mask_b = (r==V), (g==V), (b==V)
#         h[mask_b] = 4.0 + (r-g)[mask_b]/diff[mask_b]
#         h[mask_g] = 2.0 + (b-r)[mask_g]/diff[mask_g]
#         h[mask_r] = ((g-b)[mask_r]/diff[mask_r])%6
#         return torch.cat([color_sen*(diff/(V+eps))*torch.cos(2*pi*(h/6.0)), 
#                           color_sen*(diff/(V+eps))*torch.sin(2*pi*(h/6.0)), V], 1), curr_k

#     def PHVIT(self, img, k=None):
#         H, V, I = img[:, 0:1], img[:, 1:2], img[:, 2:3]
#         H, V, I = torch.clamp(H, -1, 1), torch.clamp(V, -1, 1), torch.clamp(I, 0, 1)
#         ck = k if k is not None else self.this_k
#         if ck is None: ck = torch.full((img.size(0), 1), 0.2, device=img.device)
        
#         base = torch.clamp(torch.sin(torch.clamp(I, 0, 1) * 0.5 * pi), min=1e-4, max=1.0)
#         color_sen = base.pow(ck.view(-1, 1, 1, 1))
        
#         H_res, V_res = torch.clamp(H/(color_sen+1e-6), -1, 1), torch.clamp(V/(color_sen+1e-6), -1, 1)
#         h, s = (torch.atan2(V_res, H_res)/(2*pi))%1, torch.clamp(torch.sqrt(H_res**2+V_res**2+1e-8), 0, 1)
#         hi = torch.floor(h*6.0)
#         f, p = h*6.0-hi, I*(1.-s)
#         q, t = I*(1.-f*s), I*(1.-(1.-f)*s)
#         r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
#         res = [(I,t,p), (q,I,p), (p,I,t), (p,q,I), (t,p,I), (I,p,q)]
#         for i in range(6):
#             m = (hi==i)
#             r[m], g[m], b[m] = res[i][0][m], res[i][1][m], res[i][2][m]
#         return torch.cat([r, g, b], dim=1)