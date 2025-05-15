# pvt模块的参数按照pvt_v2_b1的参数设置

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import numbers


class TransformerBlock(nn.Module):#FDCAM(FDCS)
    def __init__(self, dim, dim2,num_heads=1, ffn_expansion_factor=None, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.conv = nn.Conv2d(dim2, dim, kernel_size=3, stride=1, padding=1)
        self.cov = nn.Conv2d(2 * dim, dim, 1)


    def forward(self,ir, vi, ir2,vi2):

        if ir.shape[2] == ir2.shape[2]:
            ir_end = ir
        else:
            ir2 = self.conv(ir2)
            ir2 = F.upsample(ir2, size=(ir.shape[2], ir.shape[3]), mode='nearest', align_corners=None)
            ir_end = ir2 + ir
        if vi.shape[3] == vi2.shape[3]:
            vi_end = vi
        else:
            vi2 = self.conv(vi2)
            vi2 = F.upsample(vi2, size=(ir.shape[2], ir.shape[3]), mode='nearest', align_corners=None)
            vi_end = vi2 + vi



        maxpool = F.max_pool2d(torch.cat([ir_end, vi_end], dim=1), kernel_size=2, stride=2)  # 使用 F.max_pool2d 函数
        avgpool = F.avg_pool2d(torch.cat([ir_end, vi_end], dim=1), kernel_size=2, stride=2)  # 使用 F.avg_pool2d 函数
        tot = maxpool + avgpool
        tot =self.cov(tot)
        tot1 = F.interpolate(tot, size=(ir.shape[2], ir.shape[3]), mode="bilinear", align_corners=False)
        act = F.sigmoid(tot1)
        d_ir = ir_end * act - ir_end
        d_vi = vi_end * act - vi_end
        ir = ir_end + d_vi
        vi = vi_end + d_ir
        x1 = self.cov(torch.cat((ir, vi), 1))


        return x1


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = torch.tensor([[1., 0., -1.],
                                     [2., 0., -2.],
                                     [1., 0., -1.]])
        self.sobel = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.sobel.weight.data.copy_(sobel_filter.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

    def forward(self, x):
        sobel = self.sobel(x)
        x = torch.abs(sobel)
        return x

class slt(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(slt, self).__init__()
        self.sobel = Sobelxy(channels)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.conv3x3_leakyrelu = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv1= nn.Conv2d(2*channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.sobel(x)
        x2 = self.conv3x3_leakyrelu(x)
        x2 = self.conv3x3_leakyrelu(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        return x


class pure(nn.Module):#TEM
    def __init__(self,dim1, dim2,init=0.01):
        super(pure, self).__init__()

        self.pre_process = nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1)
        self.pre_process1 = nn.Conv2d(dim2, dim2, kernel_size=3, stride=1, padding=1)
        self.maxpoolh = nn.AdaptiveMaxPool2d(1)
        self.avgpoolh = nn.AdaptiveAvgPool2d(1)

        self.act = nn.Sigmoid()
        self.sltfeature = slt(dim2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(dim1, dim2, kernel_size=1)
        self.conv1 = nn.Conv2d(dim2, dim2, kernel_size=1)
        self.conv3x3_leakyrelu = nn.Sequential(
            nn.Conv2d(dim2, dim2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x, y):

        x1 = self.conv(x)

        x2 = self.avgpool(x1)
        y2 = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        y4 = self.avgpool(y2)
        x3 = self.avgpoolh(x1)
        y1 = self.avgpoolh(y)


        m2=x2+y4
        m3 = x1 + y2
        m3h = F.interpolate(m2, size=(m3.shape[2], m3.shape[3]), mode="bilinear", align_corners=False)
        m3 = m3h +m3
        m4h = F.interpolate(m3, size=(y.shape[2], y.shape[3]), mode="bilinear", align_corners=False)
        m4 = m4h+y+x3+y1

        #x=self.pre_process(m4)
        x = self.conv3x3_leakyrelu(m4)
        xdp = self.pre_process1(x)
        x_max = xdp * self.act(self.maxpoolh(xdp))
        x_s = self.sltfeature(x)
        out = x_max+ x +x_s
        return out




