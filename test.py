# -*- coding:utf-8 -*-
import os
import warnings

from PIL import Image

from datasets import ImageDataset

"""

-i "https://mirrors.bfsu.edu.cn/pypi/web/simple/"

"""
from arg_fusion import test_cfg
import torch
from torch.utils import data as Data
import utils
from arg_fusion import args
# from net_11 import fusion_auto
from last.last25 import system
import numpy as np
import arg_fusion

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore")



def init_net(opt):
    net = system()
    test_dataset = ImageDataset(opt, mode='image_test')
    test_dataloader = Data.DataLoader(
        test_dataset, shuffle=False, batch_size=opt.test_batch_size
    )
    para = sum([np.prod(list(p.size())) for p in net.parameters()])
    type_size = 4
    print('Model CMEFusion : params: {:4f}M'.format(para * type_size / 1000 / 1000))
    print('Model Name:MMFusion25TNO')
    return net, test_dataloader


def eval(opt, net, test_dataloader):
    path = args.model_default
    net.load_state_dict(torch.load(path))
    # for name, param in net.named_parameters():
    #     print(name, "\t", param.size())
    #     print(param)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net = net.to(device)
    net.eval()
    output_path_root = 'outputs/MMFusion25TNO/'

    if os.path.exists(output_path_root) is False:
        os.mkdir(output_path_root)
    output_count = 0
    for step, data in enumerate(test_dataloader, 1):
        vis_img = data['VIS'].float().to(device)
        inf_img = data['IR'].float().to(device)
        inf_img = inf_img.permute(0, 1, 2, 3)  # B C H W
        vis_img = vis_img.permute(0, 1, 2, 3)  # B C H W
        inf_img = inf_img.view(inf_img.shape[0], 1, *inf_img.shape[2:])
        vis_img = vis_img.view(vis_img.shape[0], 1, *inf_img.shape[2:])
        fused_img = net(inf_img, vis_img)

        print(fused_img.shape)
        file_name = str(output_count).zfill(3) + '.png'
        # file_name = os.path.basename(vis_img)+ '.png'
        output_path = output_path_root + file_name
        output_count += 1
        utils.save_image_test(fused_img, output_path)

#
# def eval(opt, net, test_dataloader):
#     path = args.model_default
#     net.load_state_dict(torch.load(path))
#     # for name, param in net.named_parameters():
#     #     print(name, "\t", param.size())
#     #     print(param)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # device = torch.device('cpu')
#     net = net.to(device)
#     net.eval()
#     output_path_root = 'outputs/CMEFusion11_epoch241_MSRS/'
#
#     if os.path.exists(output_path_root) is False:
#         os.mkdir(output_path_root)
#     output_count = 1
#     for step, data in enumerate(test_dataloader, 1):
#         vis_img = data['VIS'].float().to(device)
#         inf_img = data['IR'].float().to(device)
#         inf_img = inf_img.permute(0, 1, 2, 3)  # B C H W
#         vis_img = vis_img.permute(0, 1, 2, 3)  # B C H W
#         inf_img = inf_img.view(inf_img.shape[0], 1, *inf_img.shape[2:])
#         vis_img = vis_img.view(vis_img.shape[0], 1, *inf_img.shape[2:])
#         fused_img = net(inf_img, vis_img)
#
#         print(fused_img.shape)
#         input_folder_path = "E:\\BackBone\\image_test\\VIS_RGB\\"
#         for filename in os.listdir(input_folder_path):
#             input_filepath = os.path.join(input_folder_path, filename)
#         # file_name = str(output_count).zfill(3) + '.png'
#         #file_name = os.path.basename(vis_img)+ '.png'
#             output_path = output_path_root + filename
#
#         output_count += 1
#         utils.save_image_test(fused_img, output_path)
#
# #



#
#
# def load_model(path):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 第一行代码
#     model = fusion_auto()
#     model.load_state_dict(torch.load(path))
#     return model.to(device)
#
#
#
#
# def run_demo(nest_model, infrared_path, visible_path, output_path_root, index):
#
#     img_ir, h, w, c = utils.get_test_image(infrared_path)
#     img_vi, h, w, c = utils.get_test_image(visible_path)
#     img_ir = img_ir.view(img_ir.size(0), 1, img_ir.size(2), img_ir.size(3))
#     img_vi = img_vi.view(img_vi.size(0), 1, img_vi.size(2), img_vi.size(3))
#     if c == 1:
#         if args.cuda:
#             img_ir = img_ir.cuda()
#             img_vi = img_vi.cuda()
#         img_ir = Variable(img_ir, requires_grad=False)
#         img_vi = Variable(img_vi, requires_grad=False)
#         # encoder
#         img_fusion_list = nest_model(img_ir, img_vi)
#
#     else:
#         # fusion each block
#         img_fusion_blocks = []
#         for i in range(c):
#             # encoder
#             img_vi_temp = img_vi[i]
#             img_ir_temp = img_ir[i]
#             if args.cuda:
#                 img_vi_temp = img_vi_temp.cuda()
#                 img_ir_temp = img_ir_temp.cuda()
#             img_vi_temp = Variable(img_vi_temp, requires_grad=False)
#             img_ir_temp = Variable(img_ir_temp, requires_grad=False)
#             img_fusion_list = nest_model(img_ir_temp, img_vi_temp)
#
#     ############################ multi outputs ##############################################
#     output_count = 0
#     for img_fusion in img_fusion_list:
#         file_name = str(index).zfill(2) + '.png'
#         # file_name = ' str(index)'.zfill(2) + '.png'
#         output_path = output_path_root + file_name
#         output_count += 1
#         # save images
#         utils.save_image_test(img_fusion, output_path)
#         # print(output_path)


def rotate_images_in_folder(folder_path):
    # 获取文件夹中所有文件名
    file_names = os.listdir(folder_path)

    # 遍历文件夹中的每个文件
    for file_name in file_names:
        # 检查文件是否为图片文件（你可以根据需要调整支持的图片格式）
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, file_name)

            # 打开图像文件
            img = Image.open(file_path)

            # 逆时针旋转90度
            rotated_img = img.rotate(90, expand=True)

            # 保存旋转后的图像
            rotated_img.save(file_path)

            # 关闭图像文件
            img.close()

# def main():
#     # run demo
#
#     test_path = "F:\\CODE\\Image-Fusion-main\\Image-Fusion-main\\General Evaluation Metric\\Image\\Source-Image\\TNO"
#
#     deepsupervision = False
#
#     with torch.no_grad():
#
#         model_path = args.model_default
#         model = load_model(model_path)
#         output_path = './outputs/test003/'
#         if os.path.exists(output_path) is False:
#             os.mkdir(output_path)
#         output_path = output_path + '/'
#
#         print('Processing......  ')
#
#         for i in range(42):
#             i = i + 1
#             # infrared_path = test_path + 'IR' + '/(' + str(index) + ').png'  #
#             infrared_path = os.path.join(test_path, 'ir', f'{i:02d}.png')
#             # visible_path = test_path + 'VIS' + '/(' + str(index) + ').png'
#             visible_path = os.path.join(test_path, 'vi', f'{i:02d}.png')
#             # infrared_path = test1_path + '(' + str(index) + ').png'
#             # visible_path = test2_path + '(' + str(index) + ').png'
#             run_demo(model, infrared_path, visible_path, output_path, i)
#     print('Done......')
#
#
# if __name__ == '__main__':
#
#     # if torch.cuda.is_available():
#     #     print("CUDA is available.")
#     # else:
#     #     print("CUDA is not available.")
#     #     # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     # main()
#     folder_path = "F:\\CODE\\CMEFusion\\outputs\\epoch_51"
#     rotate_images_in_folder(folder_path)


if __name__ == '__main__':
    opt = test_cfg()
    with torch.no_grad():
        net, test_dataloader = init_net(opt)
        eval(opt, net, test_dataloader)