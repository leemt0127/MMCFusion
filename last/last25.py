import torch
import torch.nn as nn
import torch.nn.functional as F
from last.pvt1 import PyramidVisionTransformerV2, pvt_v2_b1,PyramidVisionTransformerV2_one
from last.interact25 import TransformerBlock, pure



class system(nn.Module):
    def __init__(self, in_chans=1, hidden_chans=[64, 128, 320, 512], pool_ratio=[8, 6, 4, 2], out_chans=1, linear=True):
        super(system, self).__init__()

        self.pool_ratio = pool_ratio
        self.pre_x = nn.Conv2d(in_chans, hidden_chans[0], 3, 1, 1)
        self.pre_y = nn.Conv2d(in_chans, hidden_chans[0], 3, 1, 1)
        self.pure4 = pure(hidden_chans[3],hidden_chans[2])
        self.pure3 = pure(hidden_chans[2],hidden_chans[1])
        self.pure2 = pure(hidden_chans[1],hidden_chans[0])
        self.pure1 = pure(hidden_chans[0],hidden_chans[0])
        self.un_x = pvt_v2_b1()
        self.un_y = pvt_v2_b1()
        self.Tanh2 = nn.Tanh()
        self.fuse0 = TransformerBlock(hidden_chans[0],hidden_chans[1])
        self.fuse1 = TransformerBlock(hidden_chans[1],hidden_chans[2])
        self.fuse2 = TransformerBlock(hidden_chans[2],hidden_chans[3])
        self.fuse3 = TransformerBlock(hidden_chans[3],hidden_chans[3])

        self.last = nn.Sequential(
            nn.Conv2d(hidden_chans[0], out_chans, kernel_size=3,stride=1,padding=1)
        )

    # x:ir,y:vi
    def forward(self, x, y):
        h, w = x.shape[2], x.shape[3]

        img_size = (h, w)
        x = self.pre_x(x)
        y = self.pre_y(y)
        short_x = x
        short_y = y
        x = self.un_x(x)
        y = self.un_y(y)

        fuse0 = self.fuse0(x[0], y[0],x[1], y[1])
        fuse1 = self.fuse1(x[1], y[1],x[2], y[2])
        fuse2 = self.fuse2(x[2], y[2],x[3], y[3])
        fuse3 = self.fuse3(x[3], y[3],x[3], y[3])


        out3 = self.pure4(fuse3,fuse2)
        out2 = self.pure3(out3,fuse1)
        out1 = self.pure2(out2,fuse0)
        out = self.pure1(out1,short_x+short_y)
        out = self.last(out)
        out = self.Tanh2(out)

        return out






print("----------------------------last1-------------------------------------")
a = torch.randn(1, 1, 280, 280)
b = torch.randn(1, 1, 280, 280)

model = system()
out = model(a, b)
model = system()
total = sum([param.nelement() for param in model.parameters()])
print("输出大小：{} 参数量：{} x 1e6".format(out.shape, total / 1000000))
