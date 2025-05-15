import numpy as np
import torch
import torch.nn.functional as F
import torchvision
# from skimage.filters.rank import entropy
# from minpy import numpy as np
# from skimage.morphology import disk

'''
极小项的取值究竟该如何自处？
'''
"""
在net—6 中，使用空间通道融合策略，同时极小值设置为1e-4，进行实验。
"""

EPSILON = 1e-4
import os


def attention_fusion_weight(tensor1, tensor2):
    f_channel = channel_fusion(tensor1, tensor2)
    # print('f_channel', f_channel.grad_fn)
    f_spatial = spatial_fusion(tensor1, tensor2)
    # print('f_spatial', f_spatial.grad_fn)
    # f_en = en_fusion(tensor1, tensor2)
    # tensor_f = (f_channel + f_spatial + f_en) / 3
    # tensor_f = (f_channel + f_spatial) / 2/
    tensor_f = (f_channel + f_spatial)/2
    # tensor_f = f_spatial
    return tensor_f



def channel_fusion(tensor1, tensor2):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1)
    global_p2 = channel_attention(tensor2)

    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()

    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# def en_fusion(input1, input2, spatial_type='mean'):
#     tensor1 = input1.view(input1.size(0), -1).to(torch.float32)
#     tensor2 = input2.view(input2.size(0), -1).to(torch.float32)
#     B, C, H, W = input1.shape
#     """
#     修改：将张量变换类型
#     """
#     # tensor1 = tensor1.to(torch.float32)
#     # tensor2 = tensor2.to(torch.float32)
#     # shape = tensor1.size()
#
#     # tensor1 = tensor1.sum(dim=1, keepdim=True)
#     # tensor2 = tensor2.sum(dim=1, keepdim=True)
#
#     tensor1 = (tensor1 - torch.min(tensor1)) / (torch.max(tensor1) - torch.min(tensor1)+EPSILON)
#     tensor2 = (tensor2 - torch.min(tensor2)) / (torch.max(tensor2) - torch.min(tensor2)+EPSILON)
#     """
#     这里进行了一个代码修改
#     加入了.detach()
#     """
#     spatial1 = entropy(tensor1.detach().cpu().numpy(), disk(7)).astype(np.float32)
#     spatial2 = entropy(tensor2.detach().cpu().numpy(), disk(7)).astype(np.float32)
#
#     spatial1 = torch.from_numpy(spatial1).cuda()
#     spatial2 = torch.from_numpy(spatial2).cuda()
#
#     # get weight map, soft-max
#     en_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
#     en_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
#
#     en_w1 = en_w1.view(en_w1.size(0), -1, 1, 1)
#     en_w2 = en_w2.view(en_w2.size(0), -1, 1, 1)
#     en_w1 = en_w1.view(B, C, H, W)
#     en_w2 = en_w2.view(B, C, H, W)
#     # en_w1, input1 = torch.broadcast_tensors(en_w1, input1)
#     # en_w2, input2 = torch.broadcast_tensors(en_w2, input2)
#
#     tensor_f = en_w1 * input1 + en_w2 * input2
#
#     return tensor_f
#

# channel attention
def channel_attention(tensor):
    pooling_type = 'avg'
    # global pooling
    shape = tensor.size()
    pooling_function = F.avg_pool2d

    if pooling_type == 'attention_avg':
        pooling_function = F.avg_pool2d
    elif pooling_type == 'attention_max':
        pooling_function = F.max_pool2d
    elif pooling_type == 'attention_nuclear':
        pooling_function = nuclear_pooling

    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p


"""
    channel_attention 函数实现通道注意力机制，用于计算输入张量在通道维度上的注意力权重。
    该函数通过全局池化函数（如平均池化、最大池化或核范数池化）将输入张量的高和宽维度进行降维，
    得到一个形状为 (batch_size, num_channels, 1, 1) 的全局池化结果。
"""


# spatial attention
def spatial_attention(tensor, spatial_type='mean'):
    spatial = []
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)

    return spatial


"""
    spatial_attention 函数实现空间注意力机制，用于计算输入张量在空间维度上的注意力权重。
    该函数通过对输入张量在通道维度上进行池化（如求均值或求和），得到一个形状为 (batch_size, 1, height, width)
    的空间池化结果。

"""


# pooling function


def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors


"""
    nuclear_pooling 函数实现核范数池化，用于在空间维度上对输入张量的特征图进行降维。
    该函数通过对每个特征图进行奇异值分解（SVD）得到奇异值之和，并将所有特征图的奇异值之和合并成一个形状为 
    (batch_size, num_channels, 1, 1) 的向量。

"""


def imshow(img, name, type):
    # global path
    img = torch.squeeze(img, dim=0)
    nrow = img.size(0)

    img = torchvision.utils.make_grid(img, nrow=nrow, normalize=True, padding=2)

    for i in range(nrow):
        path = './plt_png/' + str(name) + str(type)
        imgname = str(i) + '.jpg'
        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(img[i], os.path.join(path, imgname))


"""
    这是一个函数，用于将输入的图像张量可视化并保存到磁盘上。具体而言，它接受三个参数：
    img: 输入的图像张量，形状为 (batch_size, num_channels, height, width)。
    name: 保存图像的文件名前缀，用于区分不同的图像序列。
    type: 保存图像的文件名后缀，用于指定保存的图像格式（如 '.jpg', '.png' 等）。
    该函数首先将输入的图像张量通过 torchvision.utils.make_grid 函数转换为一个形状为 (3, H, W) 的三通道图像，
    其中 H 和 W 分别表示每个小图像的高度和宽度。然后，它将每个小图像都保存到磁盘上，以便后续查看和分析。
    每个小图像的保存路径由 path 和 imgname 参数指定，其中 path 表示保存文件的文件夹路径，imgname 表示保存文件的文件名，
    它由小图像的索引号和文件名后缀组成。最后，该函数通过 torchvision.utils.save_image 函数将每个小图像保存到磁盘上。
"""
