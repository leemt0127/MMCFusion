import torch.nn.functional as F
import torch
from math import exp
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ir_loss(fused_result, input_ir):
    a = fused_result - input_ir
    b = torch.square(fused_result - input_ir)
    c = torch.mean(torch.square(fused_result - input_ir))
    ir_loss = c
    return ir_loss


def vi_loss(fused_result, input_vi):
    vi_loss = torch.mean(torch.square(fused_result - input_vi))
    return vi_loss


def ssim_loss(fused_result, input_ir, input_vi):
    ssim_ = SSIM()
    ssim_loss = SSIM(fused_result, torch.maximum(input_ir, input_vi))

    return ssim_loss


def gra_loss(fused_result, input_ir, input_vi):
    gra_loss = torch.norm(Gradient(fused_result) - torch.maximum(Gradient(input_ir), Gradient(input_vi)))
    return gra_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window



def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / (((mu1_sq + mu2_sq + C1) * v2)+1e-6)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

def ssim_b(im_1, im_2):
    assert im_1.shape == im_2.shape
    b, c, w, h = im_1.shape
    ssim_ = 0.0
    for i in range(b):
        ssim_ += ssim(im_1[i, :, :, :].reshape(c, w, h), im_2[i, :, :, :].reshape(c, w, h))
        # print(im_1[i, :, :, :].shape)
    return ssim_ / b


class gradient(nn.Module):
    def __init__(self):
        super(gradient, self).__init__()
        x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        x_kernel = torch.FloatTensor(x_kernel).unsqueeze(0).unsqueeze(0)
        y_kernel = torch.FloatTensor(y_kernel).unsqueeze(0).unsqueeze(0)
        self.x_weight = nn.Parameter(data=x_kernel, requires_grad=False)
        self.y_weight = nn.Parameter(data=y_kernel, requires_grad=False)

    def forward(self, input):
        input = input.view(input.size(0), 1, input.size(-2), input.size(-1))
        x_grad = torch.nn.functional.conv2d(input, self.x_weight, padding=1)
        y_grad = torch.nn.functional.conv2d(input, self.y_weight, padding=1)
        gradRes = torch.mean((x_grad + y_grad).float())
        return gradRes


def Gradient(x):
    gradient_model = gradient().to(device)
    g = gradient_model(x)
    return g

class Gradient_Loss(nn.Module):
    def __init__(self):
        super(Gradient_Loss, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x, y):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient_1 = torch.abs(grad_x) + torch.abs(grad_y)

        grad_x = F.conv2d(y, self.weight_x)
        grad_y = F.conv2d(y, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return (gradient_1 - gradient).abs().mean()

# if __name__ == '__main__':
#     x = torch.rand([4, 1, 256, 256])
#     x = x.cuda()
#     model = Gradient
#     y = model(x)
#     print(y)
