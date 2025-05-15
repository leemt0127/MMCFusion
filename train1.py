import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
from torch.utils import data as Data
import torch
from torch.optim import Adam
from torch.autograd import Variable
from utils import LambdaLR
from arg_fusion import args, train_cfg

import utils
import warnings
from loss_all import SSIM, Gradient_Loss
import sys
import numpy as np
from datasets import ImageDataset
import cv2

warnings.filterwarnings("ignore")
"""
每次做实验板数记得修改这个位置
"""
# from net_5 import fusion_auto
from last.last26 import system
device = 'cuda'
'''
14lossgai修改了三个损失的参数分别为1,10,100,和学习率退火改为指数(学习率没变）
14lossgai2修改了ssim函数
14lossgai3恢复ssim,修改grad（失败）、
14lossgai3修改了ssim函数
14lossgai4修改了grad函数的系数为0.5
14lossgai4修改了输入图像为256,MSRS_train_patch->MSRS_train_256
'''


def main():
    # 首先进行数据预处理，确定双模态patches维度都是256x256的
    # 然后确定训练样本数量
    # 这版实验是128x128
    opt = train_cfg()
    '''
    还有这里
    '''
    train(opt, 'MMFusion262')


def train(opt, name):
    test_dataset = ImageDataset(opt, mode='test')
    test_dataloader = Data.DataLoader(
        test_dataset, shuffle=False, batch_size=opt.test_batch_size
    )
    train_dataset = ImageDataset(opt, mode='train')
    train_dataloader = Data.DataLoader(
        train_dataset, shuffle=False, batch_size=opt.batch_size
    )
    model = system()
    # for name, param in model.named_parameters():
    #     print('name:{} param grad:{} param requires_grad:{}'.format(name, param.grad, param.requires_grad))

    optimizer = Adam(model.parameters(), opt.lr)
    # optimizer = torch.optim.adagrad(model.parameters(), lr=opt.lr)
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_epochs).step)
    # lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    pixel_loss_inf = torch.nn.L1Loss()
    pixel_loss_vis = torch.nn.L1Loss()
    ssim_loss = SSIM()
    grad_loss = Gradient_Loss()
    model.cuda()
    save_loss = float('inf')
    train_batch_all = len(train_dataloader)
    test_batch_all = len(test_dataloader)
    train_loss_ch = float('inf')
    test_loss_ch = float('inf')
    for epoch in range(0, opt.n_epochs):
        # logger_train = Logger(args.epochs, len(train_dataloader), width=256, height=256)
        # logger_val = Logger(args.epochs, len(test_dataloader), mode='Val', width=256, height=256)
        model.train()
        torch.cuda.empty_cache()  # 释放显存
        count = 0
        train_loss, train_ssim, train_psnr, train_mi, train_en, train_batch, train_grad, train_pixel = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        time_start_train = time.time()
        time_start_test = time.time()
        for i, batch in enumerate(train_dataloader):
            inf_img = batch['IR'].float().to(device)
            vis_img = batch['VIS'].float().to(device)
            count += 1
            optimizer.zero_grad()
            outputs = model(inf_img, vis_img)
            # logger_train.log(
            #     images={'visible_image': vis_img, 'infrared_image': inf_img, 'fused-image': outputs})
            pixel_loss_value = pixel_loss_vis(outputs, vis_img) + pixel_loss_inf(outputs, inf_img)
            # ssim_loss_value = 1 - ssim_loss(outputs, torch.maximum(inf_img, vis_img))
            ssim_loss_value = 1 - ssim_loss(outputs, torch.maximum(inf_img, vis_img))
            grad_loss_value = 0.5 * (grad_loss(outputs, torch.maximum(inf_img, vis_img)))
            # grad_loss_value = 0.5 * (grad_loss(outputs, vis_img) + grad_loss(outputs, inf_img))
            total_loss = pixel_loss_value + 100 * grad_loss_value
            total_loss.requires_grad_(True)
            total_loss.backward(create_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_batch += 1
            train_loss += total_loss.item()
            train_pixel += pixel_loss_value.item()
            train_grad += grad_loss_value.item()
            sys.stdout.write(
                '\rTrain Epoch %03d/%03d [%04d/%04d] - all_train_loss: %.6f - '
                'ssim_loss: %.6f - pixel_loss: %.6f - grad_loss: %.6f  '
                '- train_time: %.6f - ' %
                (epoch + 1, opt.n_epochs, train_batch, train_batch_all, (train_loss / train_batch),
                 ssim_loss_value, pixel_loss_value, grad_loss_value,
                 time.time() - time_start_train))

        sys.stdout.write('\n')
        lr_scheduler_G.step()
#测试注销部分
        with torch.no_grad():
            model.eval()
            test_all_loss, test_ssim, test_batch, test_pixel_loss, test_ssim_loss, test_grad_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            test_EN, test_MI, test_SF, test_SD, test_AG, test_PSNR, test_MSE, test_SCD, test_VIF, test_CC = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # -----------------------------
            # 还有这里
            # -----------------------------

            output_path_root = r'outputs/MMFusion262/epoch_' + str(epoch + 1) + '/'
            if os.path.exists(output_path_root) is False:
                # print(os.path.exists(output_path_root))
                os.makedirs(output_path_root)
            output_count = 1
            for i, patch in enumerate(test_dataloader):
                vis_img = patch['VIS'].float().to(device)
                inf_img = patch['IR'].float().to(device)
                # inf_img = Variable(inf_img, requires_grad=False)
                # vis_img = Variable(vis_img, requires_grad=False)
                inf_img = inf_img.permute(0, 1, 2, 3)  # B C H W
                vis_img = vis_img.permute(0, 1, 2, 3)  # B C H W
                test_outputs = model(inf_img, vis_img)
                file_name = str(output_count).zfill(2) + '.png'
                output_path = output_path_root + file_name
                output_count += 1
                # save images
                utils.save_image_test(test_outputs, output_path)

                # logger_val.log(
                #     images={'visible_image': vis_img, 'infrared_image': inf_img, 'fused-image': test_outputs})
                pixel_loss_value = pixel_loss_vis(test_outputs, vis_img) + pixel_loss_inf(test_outputs, inf_img)
                ssim_loss_value = 1 - ssim_loss(test_outputs, torch.maximum(inf_img, vis_img))
                # grad_loss_value = grad_loss(test_outputs, torch.maximum(inf_img, vis_img))
                grad_loss_value = 0.5 * (grad_loss(test_outputs, torch.maximum(inf_img, vis_img)))
                test_loss = pixel_loss_value + 100 * grad_loss_value
                ssim_step = 1 - ssim_loss_value.item()
                test_ssim_loss += ssim_loss_value.item()
                test_pixel_loss += pixel_loss_value.item()
                test_grad_loss += grad_loss_value.item()
                test_all_loss += test_loss.item()
                test_ssim += ssim_step
#注销结束





                # x_vis = vis_img.cpu().detach().numpy()
                # x_inf = inf_img.cpu().detach().numpy()
                # test_outputs = test_outputs.cpu().detach().numpy()
                #
                # f_img_int = np.array(test_outputs).astype(np.int32)
                # f_img_double = np.array(test_outputs).astype(np.float32)
                #
                # ir_img_int = np.array(x_inf).astype(np.int32)
                # ir_img_double = np.array(x_inf).astype(np.float32)
                #
                # vi_img_int = np.array(x_vis).astype(np.int32)
                # vi_img_double = np.array(x_vis).astype(np.float32)
                #
                # EN = EN_function(f_img_int)
                # MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)
                # SF = SF_function(f_img_double)
                # SD = SD_function(f_img_double)
                # AG = AG_function(f_img_double)
                # PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)
                # MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)
                # VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)
                # CC = CC_function(ir_img_double, vi_img_double, f_img_double)
                # SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)
                #
                # # --------------------起飞喽------------------
                #
                # test_EN += EN
                # test_MI += MI
                # test_SF += SF
                # test_SD += SD
                # test_AG += AG
                # test_PSNR += PSNR
                # test_MSE += MSE
                # test_VIF += VIF
                # test_CC += CC
                # test_SCD += SCD
        #测试注销部分
                test_batch += 1

                sys.stdout.write(
                    '\rTest Epoch %03d/%03d [%04d/%04d] - ' % (epoch + 1, opt.n_epochs, test_batch, test_batch_all))
                sys.stdout.write('test_loss: %.6f ' % (test_all_loss / test_batch))

        #结束
                # sys.stdout.write('test_EN: %.6f - ' % (test_EN / test_batch))
                # sys.stdout.write('test_MI: %.6f - ' % (test_MI / test_batch))
                # sys.stdout.write('test_SF: %.6f - ' % (test_SF / test_batch))
                # sys.stdout.write('test_SD: %.6f - ' % (test_SD / test_batch))
                # sys.stdout.write('test_AG: %.6f - ' % (test_AG / test_batch))
                # sys.stdout.write('test_PSNR: %.6f - ' % (test_PSNR / test_batch))
                # sys.stdout.write('test_MSE: %.6f - ' % (test_MSE / test_batch))
                # sys.stdout.write('test_VIF: %.6f  ' % (test_VIF / test_batch))
                # sys.stdout.write('test_CC: %.6f  ' % (test_CC / test_batch))
                # sys.stdout.write('test_SCD: %.6f  ' % (test_SCD / test_batch))

        sys.stdout.write('\n')
        sys.stdout.write('==' * 100)
        sys.stdout.write('\n')

        if not os.path.exists('checkpoint/' + name + '/'):
            os.makedirs('checkpoint/' + name + '/')

        if train_loss_ch > train_loss / train_batch:
            torch.save(model.state_dict(), 'checkpoint/' + name + '/train_net_' + str(epoch + 1) + '.pth')
            train_loss_ch = train_loss / train_batch
        #测试注销部分
        if test_loss_ch > test_all_loss / test_batch:
            torch.save(model.state_dict(), 'checkpoint/' + name + '/test_net_' + str(epoch + 1) + '.pth')
            test_loss_ch = test_all_loss / test_batch

        if epoch == opt.n_epochs - 1:
            torch.save(model.state_dict(), 'checkpoint/' + name + '/final_epoches.pth')


if __name__ == "__main__":
    main()
