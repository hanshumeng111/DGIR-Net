# import torch
# import torch.nn as nn
# from net.HVI_transform import RGB_HVI
# from net.transformer_utils import *
# from net.DGDI import *
# from huggingface_hub import PyTorchModelHubMixin

# class DGIRNet(nn.Module, PyTorchModelHubMixin):
#     def __init__(self,
#                  channels=[36, 36, 72, 144],
#                  heads=[1, 2, 4, 8],
#                  norm=False
#                  #解析通道数，注意力头数，和归一化标志
#         ):
#         super(DGIRNet, self).__init__()
#         [ch1, ch2, ch3, ch4] = channels
#         [head1, head2, head3, head4] = heads

#         # HV_ways
#         #编码器 输入层 HVI的3通道->ch1通道（36）
#         self.HVE_block0 = nn.Sequential(
#             nn.ReplicationPad2d(1), #复制填充，避免边缘信息丢失
#             nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False) #（输入通道，输出通道，核数）
#             )
#         #三次下采样 从ch1到ch4
#         self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
#         self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
#         self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
#         #解码器 上采样 恢复尺寸 输出增强后的HV色彩通道
#         self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
#         self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
#         self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
#         self.HVD_block0 = nn.Sequential(
#             nn.ReplicationPad2d(1),
#             nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
#         )


#         # I_ways
#         #编码器（下采样）：从HVI的I通道（1通道）提取强度特征
#         self.IE_block0 = nn.Sequential(
#             nn.ReplicationPad2d(1),
#             nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
#             )
#         #三次下采样
#         self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
#         self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
#         self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

#         #上采样 恢复尺寸 输出增强后的I通道
#         self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
#         self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
#         self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
#         self.ID_block0 =  nn.Sequential(
#             nn.ReplicationPad2d(1),
#             nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
#             )

#         # -------------------------- DGDI模块：6个跨注意力交互（文献核心设计）--------------------------
#         self.HV_DGDI1 = HV_DGDI(ch2, head2) #尺度1（ch2=36，head2=2）
#         self.HV_DGDI2 = HV_DGDI(ch3, head3)
#         self.HV_DGDI3 = HV_DGDI(ch4, head4)
#         self.HV_DGDI4 = HV_DGDI(ch4, head4)
#         self.HV_DGDI5 = HV_DGDI(ch3, head3)
#         self.HV_DGDI6 = HV_DGDI(ch2, head2)

#         #DGDI I分支的DGDI 以HV分支特征为辅助，增强I分支特征表达
#         self.I_DGDI1 = I_DGDI(ch2, head2)
#         self.I_DGDI2 = I_DGDI(ch3, head3)
#         self.I_DGDI3 = I_DGDI(ch4, head4)
#         self.I_DGDI4 = I_DGDI(ch4, head4)
#         self.I_DGDI5 = I_DGDI(ch3, head3)
#         self.I_DGDI6 = I_DGDI(ch2, head2)

#         self.trans = RGB_HVI() #调用

#     def forward(self, x): #x:输入srgb的图像 （shape：B×3×H×W）
#         dtypes = x.dtype #输入x的类型
#         hvi = self.trans.HVIT(x) #输入RGB->输出HVI
#         i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)  # 拆分I通道：HVI的第2通道（I），作为I-branch输入
#         # low 编码器前向传播
#         i_enc0 = self.IE_block0(i) # I分支的原始1通道强度图，先提取为36通道的浅层特征（i_enc0）
#         i_enc1 = self.IE_block1(i_enc0) #再下采样为尺寸减半的特征（i_enc1）
#         hv_0 = self.HVE_block0(hvi)
#         hv_1 = self.HVE_block1(hv_0)
#         #保存跳跃连接特征
#         i_jump0 = i_enc0
#         hv_jump0 = hv_0

#         #第一次DGDI交互 （尺度1：ch2=36，head2=2）+ 下采样
#         i_enc2 = self.I_DGDI1(i_enc1, hv_1) # I-branch用HV-branch的hv_1引导，优化强度特征
#         hv_2 = self.HV_DGDI1(hv_1, i_enc1) # HV-branch用I-branch的i_enc1引导，优化色彩特征
#        #保存跳连接1（尺度1交互后的特征）
#         v_jump1 = i_enc2
#         hv_jump1 = hv_2
#         #第二次下采样：尺度从H/2->H/4
#         i_enc2 = self.IE_block2(i_enc2)  #36->72通道
#         hv_2 = self.HVE_block2(hv_2)

#         #第二次DGDI交互（尺度2：ch3=72，head3=4）+ 下采样
#         i_enc3 = self.I_DGDI2(i_enc2, hv_2)
#         hv_3 = self.HV_DGDI2(hv_2, i_enc2)
#         #第二个跳连接
#         v_jump2 = i_enc3
#         hv_jump2 = hv_3
#         #第三次下采样（最深层）
#         i_enc3 = self.IE_block3(i_enc2)
#         hv_3 = self.HVE_block3(hv_2)

#          # 第三次DGDI交互（尺度3：ch4=144，head4=8）：编码器最深层特征交互
#         i_enc4 = self.I_DGDI3(i_enc3, hv_3)
#         hv_4 = self.HV_DGDI3(hv_3, i_enc3)

#         # -------------------------- 第二阶段：解码器前向传播（上采样+DGDI交互+跳连接融合）--------------------------
#          #high 解码器初始DGDI交互（最深层）
#         i_dec4 = self.I_DGDI4(i_enc4,hv_4)
#         hv_4 = self.HV_DGDI4(hv_4, i_enc4)

#         #第一次上采样+跳连接融合DGDI交互
#         hv_3 = self.HVD_block3(hv_4, hv_jump2) # HV-branch上采样：144→72，融合跳连接2（浅层细节）
#         i_dec3 = self.ID_block3(i_dec4, v_jump2)  # I-branch上采样：144→72，融合跳连接2
#         #DGDI交互（尺度2：ch3=72）:上采样后进一步优化
#         i_dec2 = self.I_DGDI5(i_dec3, hv_3)
#         hv_2 = self.HV_DGDI5(hv_3, i_dec3)

#         #第二次上采样+跳连接融合+DGDI交互
#         hv_2 = self.HVD_block2(hv_2, hv_jump1)
#         i_dec2 = self.ID_block2(i_dec3, v_jump1)
#         # DGDI交互（尺度1：ch2=36）
#         i_dec1 = self.I_DGDI6(i_dec2, hv_2)
#         hv_1 = self.HV_DGDI6(hv_2, i_dec2)

#         #第三次上采样+跳连接融合+输出层
#         i_dec1 = self.ID_block1(i_dec1, i_jump0) # I-branch上采样：融合跳连接0，36→36 （图像省略了最浅层的跳连接）
#         i_dec0 = self.ID_block0(i_dec1) # I-branch输出：36→1通道（增强后的I强度图）
#         hv_1 = self.HVD_block1(hv_1, hv_jump0)
#         hv_0 = self.HVD_block0(hv_1) #增强后的HV色彩图

#         #拼接增强后的HV色彩图和I强度图，同时与原始HVI进行残差连接，保留原始信息
#         output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
#         #感知逆变换
#         output_rgb = self.trans.PHVIT(output_hvi)
#         return output_rgb

#     def HVIT(self,x):
#         hvi = self.trans.HVIT(x)
#         return hvi


import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.DGDI import *
from huggingface_hub import PyTorchModelHubMixin

class DGIRNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(DGIRNet, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )

        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )

        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )

        self.HV_DGDI1 = HV_DGDI(ch2, head2)
        self.HV_DGDI2 = HV_DGDI(ch3, head3)
        self.HV_DGDI3 = HV_DGDI(ch4, head4)
        self.HV_DGDI4 = HV_DGDI(ch4, head4)
        self.HV_DGDI5 = HV_DGDI(ch3, head3)
        self.HV_DGDI6 = HV_DGDI(ch2, head2)

        self.I_DGDI1 = I_DGDI(ch2, head2)
        self.I_DGDI2 = I_DGDI(ch3, head3)
        self.I_DGDI3 = I_DGDI(ch4, head4)
        self.I_DGDI4 = I_DGDI(ch4, head4)
        self.I_DGDI5 = I_DGDI(ch3, head3)
        self.I_DGDI6 = I_DGDI(ch2, head2)

        self.trans = RGB_HVI()

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)

        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)

        i_jump0 = i_enc0
        hv_jump0 = hv_0

        # ===== 核心修改：收集每一层的门控图 =====
        i_enc2, g_h2i_1 = self.I_DGDI1(i_enc1, hv_1)
        hv_2, g_i2h_1 = self.HV_DGDI1(hv_1, i_enc1)

        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3, g_h2i_2 = self.I_DGDI2(i_enc2, hv_2)
        hv_3, g_i2h_2 = self.HV_DGDI2(hv_2, i_enc2)

        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4, g_h2i_3 = self.I_DGDI3(i_enc3, hv_3)
        hv_4, g_i2h_3 = self.HV_DGDI3(hv_3, i_enc3)

        i_dec4, g_h2i_4 = self.I_DGDI4(i_enc4,hv_4)
        hv_4, g_i2h_4 = self.HV_DGDI4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)

        i_dec2, g_h2i_5 = self.I_DGDI5(i_dec3, hv_3)
        hv_2, g_i2h_5 = self.HV_DGDI5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1, g_h2i_6 = self.I_DGDI6(i_dec2, hv_2)
        hv_1, g_i2h_6 = self.HV_DGDI6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        # ========================================================
        # 终极保护层：如果是测试模式(eval.py)，不要返回元组，防止测试代码报错！
        # ========================================================
        if not self.training:
            return output_rgb

        # 如果是训练模式(train.py)，打包所有的 Gate 给外面的 Loss 用
        gate_h2i_list = [g_h2i_1, g_h2i_2, g_h2i_3, g_h2i_4, g_h2i_5, g_h2i_6]
        gate_i2h_list = [g_i2h_1, g_i2h_2, g_i2h_3, g_i2h_4, g_i2h_5, g_i2h_6]

        return output_rgb, gate_h2i_list, gate_i2h_list

    def load_state_dict(self, state_dict, strict=True):
        """Load current checkpoints and checkpoints saved before the DGDI rename."""
        renamed_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("HV_LCA", "HV_DGDI").replace("I_LCA", "I_DGDI")
            renamed_state_dict[new_key] = value
        return super().load_state_dict(renamed_state_dict, strict=strict)

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
