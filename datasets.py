import os
import numpy as np
import random
from glob import glob
import matplotlib.pyplot as plt
from arg_fusion import train_cfg
from torch.utils import data as Data
from PIL import Image
import torchvision.transforms as transforms
import torch


def im_mode(mode, image):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down 上下翻转
        return np.flipud(image)
    elif mode == 2:
        # rotate counter-wise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


class ImageDataset(Data.Dataset):
    def __init__(self, opt, mode='train'):
        self.transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.root = opt.root
        self.patch_size = opt.train_patch_size

        self.mode = mode
        random.seed(123)

        if mode == 'train':
            self.files_VIS = sorted(glob(os.path.join(self.root, 'MSRS_train_patch', 'VIS') + '/*.*'))
            self.files_IR = sorted(glob(os.path.join(self.root, 'MSRS_train_patch', 'IR') + '/*.*'))
            self.image_VIS, self.image_IR = self.getdatalist(self.files_VIS, self.files_IR)
        if mode == 'test':
            self.files_VIS = sorted(glob(os.path.join(self.root, 'test', 'VIS') + '/*.*'))
            self.files_IR = sorted(glob(os.path.join(self.root, 'test', 'IR') + '/*.*'))
            self.image_VIS, self.image_IR = self.test_data(self.files_VIS, self.files_IR)
        if mode == 'image_test':
            self.files_VIS = sorted(glob(os.path.join(self.root,'VIS_gray') + '/*.*'))
            self.files_IR = sorted(glob(os.path.join(self.root,'IR') + '/*.*'))
            self.image_VIS, self.image_IR = self.test_data(self.files_VIS, self.files_IR)
        print(len(self.image_VIS))
        print(len(self.image_IR))


    def __getitem__(self, index):
        assert len(self.files_VIS) == len(self.files_IR)
        item_VIS = self.transform(self.image_VIS[index % len(self.files_VIS)])
        item_IR = self.transform(self.image_IR[index % len(self.files_IR)])

        return {'VIS': item_VIS, 'IR': item_IR}

    def __len__(self):
        return len(self.image_VIS)

    def load_im(self, image_root):
        im = Image.open(image_root)
        return np.array(im, dtype=np.float32) / 255.0

    def getdatalist(self, name_list1, name_list2):
        image_list_x = []
        image_list_y = []
        for i in range(len(name_list1)):
            image_old_x = self.load_im(name_list1[i]).copy()
            image_old_y = self.load_im(name_list2[i]).copy()
            # h, w = image_old_x.shape
            # w_now, h_now = 0, 0
            # w_new, h_new = self.patch_size[1], self.patch_size[0]
            # while h >= h_new:
            #     while w >= w_new:
            #         image_x = image_old_x[h_now:h_now + h_new, w_now:w_new + w_now]
            #         image_y = image_old_y[h_now:h_now + h_new, w_now:w_new + w_now]
            #         w -= w_new
            #         w_now += w_new
            if self.mode == 'train':
                mode = random.randint(0, 7)
                image_x_1 = self.im_mode(mode, image_old_x).copy()
                image_y_1 = self.im_mode(mode, image_old_y).copy()
                image_list_x.append(image_x_1)
                image_list_y.append(image_y_1)
            #         else:
            #             image_list_x.append(image_x)
            #             image_list_y.append(image_y)
            #     h -= h_new
            #     h_now += h_new
            #     w = image_old_x.shape[1]
            #     w_new = self.patch_size[1]
            #     w_now = 0
            # image_list_x.append(image_old_x)
            # image_list_y.append(image_old_y)
        return image_list_x, image_list_y

    def im_mode(self, mode, image):
        if mode == 0:
            # original
            return image
        elif mode == 1:
            # flip up and down 上下翻转
            return np.flipud(image)
        elif mode == 2:
            # rotate counter-wise 90 degree
            return np.rot90(image)
        elif mode == 3:
            # rotate 90 degree and flip up and down
            image = np.rot90(image)
            return np.flipud(image)
        elif mode == 4:
            # rotate 180 degree
            return np.rot90(image, k=2)
        elif mode == 5:
            # rotate 180 degree and flip
            image = np.rot90(image, k=2)
            return np.flipud(image)
        elif mode == 6:
            # rotate 270 degree
            return np.rot90(image, k=3)
        elif mode == 7:
            # rotate 270 degree and flip
            image = np.rot90(image, k=3)
            return np.flipud(image)

    def test_data(self, x_list, y_list):
        image_list_x = []
        image_list_y = []
        for i in range(len(x_list)):
            image_list_x.append(self.load_im(x_list[i]))
            image_list_y.append(self.load_im(y_list[i]))
        return image_list_x, image_list_y


if __name__ == '__main__':
    opt = train_cfg()
    # train_dataset = ImageDataset(opt, mode='train')
    # train_dataloader = Data.DataLoader(
    #     train_dataset, shuffle=False, batch_size=1
    # )
    # print('1')
    # for step, image in enumerate(train_dataloader):
    #     image_a = image['VIS'].float()
    #     image_ay = image['IR'].float()
    #     print(torch.max(image_a))
    #     print(torch.min(image_a))
    #     print(torch.max(image_ay))
    #     print(torch.min(image_ay))
    #     print(image_a.shape)
    #     print(image_a.dtype)
    #     plt.figure(figsize=(16, 16))
    #     for i in range(4):
    #         plt.subplot(2, 4, i + 1)
    #         plt.imshow(image_a[i].numpy().transpose([1, 2, 0]))
    #         plt.axis('off')
    #         plt.subplot(2, 4, i + 5)
    #         plt.imshow(image_ay[i].numpy().transpose([1, 2, 0]))
    #         plt.axis('off')
    #     plt.show()
    #     break

    # val_dataset = ImageDataset(opt, mode='val')
    # val_dataloader = Data.DataLoader(
    #     val_dataset, shuffle=False, batch_size=4
    # )
    # for step, image in enumerate(val_dataloader):
    #     image_a = image['A']
    #     image_ay = image['AY']
    #
    #     print(image_a.shape)
    #     print(image_a.dtype)
    #     plt.figure(figsize=(16, 16))
    #     for i in range(4):
    #         plt.subplot(2, 4, i + 1)
    #         plt.imshow(image_a[i].numpy().transpose([1, 2, 0]))
    #         plt.axis('off')
    #         plt.subplot(2, 4, i + 5)
    #         plt.imshow(image_ay[i].numpy().transpose([1, 2, 0]))
    #         plt.axis('off')
    #     plt.show()
    #     break

    test_dataset = ImageDataset(opt, mode='test')
    test_dataloader = Data.DataLoader(
        test_dataset, shuffle=False, batch_size=1
    )
    for step, image in enumerate(test_dataloader):
        image_a = image['VIS']
        image_ay = image['IR']
        print(image_a.shape)
        # plt.figure(figsize=(16, 16))
        # plt.subplot(1, 2, 1)
        # plt.imshow(image_a[0].numpy().transpose([1, 2, 0]))
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(image_ay[0].numpy().transpose([1, 2, 0]))
        # plt.axis('off')
        # plt.show()
