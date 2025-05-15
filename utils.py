import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from arg_fusion import args
import matplotlib as mpl
from os import listdir
from os.path import join
from imageio import imread, imsave
from visdom import Visdom
from torchvision.transforms import functional as F
import time
import torchvision.transforms as transforms
from torch.utils import data as Data
from glob import glob
import matplotlib.pyplot as plt

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion*255
    # img_fusion = np.expand_dims(img_fusion, axis=1)

    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    img_fusion = np.squeeze(img_fusion)
    # 	img_fusion = imresize(img_fusion, [h, w])
    # img_fusion = cv2.resize(img_fusion, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_fusion_pil = Image.fromarray(img_fusion)
    img_fusion_pil = F.hflip(img_fusion_pil)
    # img_fusion_pil = F.rotate(img_fusion_pil, angle=90)
    img_fusion = np.array(img_fusion_pil)
    cv2.imwrite(output_path, img_fusion)


def imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(arr, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)


class LambdaLR():
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def step(self, epoch):
        if epoch > 10:
            return 0.95
        else:
            return 1.0


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)

    def sort_by_number(file):
        return int(''.join(filter(str.isdigit, file)))

    dir.sort(key=sort_by_number)
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
        # if file == 'COCO_train2014_000000000394.jpg':
        #     break
    return images


def tensor2image(tensor, cuda=True, isshow=False):
    if isshow:
        if cuda:
            # image = 255.0 * tensor[0].detach().cpu().numpy()
            image = 255.0 * tensor[0].cpu().detach().float().numpy()
        else:
            image = 255.0 * tensor[0].float().numpy()
    else:
        if cuda:
            image = 255.0 * tensor.cpu().detach().float().numpy()
            image = np.expand_dims(np.mean(image, axis=1), axis=1)
        else:
            image = 255.0 * tensor.float().numpy()
            image = np.expand_dims(np.mean(image, axis=1), axis=1)
    return image.astype(np.uint8)


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


class Logger():
    def __init__(self, n_epochs, batches_epoch, width, height, cuda=True, mode='Train'):
        self.n_epochs = n_epochs  # 总训练轮数
        self.batches_epoch = batches_epoch  # 总batch数
        self.epoch = 1  # epoch开始数
        self.batch = 1  # batch开始数
        self.prev_time = time.time()
        self.mean_period = 0
        self.metices = {}
        self.image_windows = {}
        self.metices_windows = {}
        self.viz = Visdom(env='mse')
        self.cuda = cuda
        self.mode = mode
        self.width = width
        self.Height = height
        if mode == 'Train':
            self.losses = {}
            self.loss_windows = {}

    def log(self, losses=None, images=None, metices=None):
        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(
                    tensor2image(tensor.data, cuda=self.cuda, isshow=True),
                    opts={'title': image_name, 'width': self.width, 'height': self.Height})
            else:
                self.viz.image(
                    tensor2image(tensor.data, cuda=self.cuda, isshow=True),
                    win=self.image_windows[image_name],
                    opts={'title': image_name, 'width': self.width, 'height': self.Height})


def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U, D, V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE):
    batches = int(len(image_path) // BATCH_SIZE)
    return image_path, batches


def get_image(path, height=256, width=256, flag=False):
    # if mode == 'L':
    #     image = imread(path, pilmode=mode)
    # elif mode == 'RGB':
    #     image = Image.open(path).convert('RGB')
    #     image = image/255.0
    #     return image
    if flag is True:
        image = imread(path)
        image = image / 255.0
    else:
        image = imread(path)
        image = image / 255.0

    if height is not None and width is not None:
        image = cv2.resize(image, (height, width))
        # image = plt.imresize(image, [height, width], interp='nearest')
    return image


# load images - test phase
def get_test_image(paths, height=None, width=None, flag=False):  # (256 256)
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = imread(path, pilmode='L')
        # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')

        base_size = 1024  #
        print(image.shape)
        h = image.shape[0]
        w = image.shape[1]
        c = 1
        if h > base_size or w > base_size:
            c = 4
            images = get_img_parts(image, h, w)
        else:
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
            images.append(image)
            images = np.stack(images, axis=0)
            images = torch.from_numpy(images).float()

    # images = np.stack(images, axis=0)
    # images = torch.from_numpy(images).float()
    return images


def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))  # 变为一半？？？
    w_cen = int(np.floor(w / 2))

    img1 = image[0:h_cen + 3, 0: w_cen + 3]  # +3  why？
    img1 = np.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])  # 增加两个维度

    img2 = image[0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])

    img3 = image[h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])

    img4 = image[h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])

    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images


def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        # save_image_test(img1, './outputs/test/block1.png')
        # save_image_test(img2, './outputs/test/block2.png')
        # save_image_test(img3, './outputs/test/block3.png')
        # save_image_test(img4, './outputs/test/block4.png')

        img_f = torch.zeros(1, 1, h, w).cuda()
        count = torch.zeros(1, 1, h, w).cuda()

        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    # img_fusion = np.expand_dims(img_fusion, axis=1)
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    img_fusion = np.squeeze(img_fusion)
    # 	img_fusion = imresize(img_fusion, [h, w])
    # img_fusion = cv2.resize(img_fusion, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_fusion_pil = Image.fromarray(img_fusion)
    img_fusion = np.array(img_fusion_pil)
    imsave(output_path, img_fusion)


def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images_ir = []
    images_vi = []
    for path in paths:
        image = get_image(path, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/ir_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_ir.append(image)

        path_vi = path.replace('lwir', 'visible')
        image = get_image(path_vi, height, width, flag)
        image = np.reshape(image, [1, height, width])
        # imsave('./outputs/vi_gray.jpg', image)
        # image = image.transpose(2, 0, 1)
        images_vi.append(image)

    images_ir = np.stack(images_ir, axis=0)
    images_ir = torch.from_numpy(images_ir).float()

    images_vi = np.stack(images_vi, axis=0)
    images_vi = torch.from_numpy(images_vi).float()
    return images_ir, images_vi


def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag=False)
        if mode == 'L':
            if image is not None:
                image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            if image is not None:
                image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = images.astype(float)
    images = torch.from_numpy(images)
    return images


def get_test_images_auto(paths):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_test_image(path)
        images.append(image)
    images = np.array(images)
    images = images.astype(float)
    images = torch.from_numpy(images)
    return images


# 自定义colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00', '#FF0000',
                                                                 '#8B0000'], 256)


"""
我就是粘贴大王
"""


class ImageDataset(Data.Dataset):
    def __init__(self, opt, mode='train'):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.root = opt.root
        self.patch_size = opt.train_patch_size
        self.mode = mode
        random.seed(123)

        if mode == 'train':
            self.files_VIS = sorted(glob(os.path.join(self.root, 'train', 'VIS') + '/*.*'))
            self.files_IR = sorted(glob(os.path.join(self.root, 'train', 'IR') + '/*.*'))
            # self.files_A = self.files_A[: int(len(self.files_A)*0.9)]
            # self.files_AY = self.files_AY[: int(len(self.files_AY)*0.9)]
            self.image_VIS, self.image_IR = self.getdatalist(self.files_VIS, self.files_IR)
        elif mode == 'test':
            self.files_VIS = sorted(glob(os.path.join(self.root, 'test', 'VIS') + '/*.*'))
            self.files_IR = sorted(glob(os.path.join(self.root, 'test', 'IR') + '/*.*'))
            self.image_VIS, self.image_IR = self.test_data(self.files_VIS, self.files_IR)
        elif mode == 'val':
            self.files_VIS = sorted(glob(os.path.join(self.root, 'train', 'VIS') + '/*.*'))
            self.files_IR = sorted(glob(os.path.join(self.root, 'train', 'IR') + '/*.*'))
            self.files_VIS = self.files_VIS[int(len(self.files_VIS) * 0.9):]
            self.files_IR = self.files_IR[int(len(self.files_AY) * 0.9):]
            self.image_VIS, self.image_IR = self.getdatalist(self.files_VIS, self.files_IR)
        print(len(self.image_VIS))
        print(len(self.image_IR))

    def __getitem__(self, index):
        assert len(self.files_VIS) == len(self.files_IR)
        item_A = self.transform(self.image_VIS[index % len(self.files_VIS)])
        item_AY = self.transform(self.image_IR[index % len(self.files_IR)])

        return {'A': item_A, 'AY': item_AY}

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
            h, w, c = image_old_x.shape
            w_now, h_now = 0, 0
            w_new, h_new = self.patch_size[1], self.patch_size[0]
            while h >= h_new:
                while w >= w_new:
                    image_x = image_old_x[h_now:h_now + h_new, w_now:w_new + w_now, :]
                    image_y = image_old_y[h_now:h_now + h_new, w_now:w_new + w_now, :]
                    w -= w_new
                    w_now += w_new
                    if self.mode == 'train':
                        mode = random.randint(0, 7)
                        image_x_1 = self.im_mode(mode, image_x).copy()
                        image_y_1 = self.im_mode(mode, image_y).copy()
                        image_list_x.append(image_x_1)
                        image_list_y.append(image_y_1)
                    else:
                        image_list_x.append(image_x)
                        image_list_y.append(image_y)
                h -= h_new
                h_now += h_new
                w = image_old_x.shape[1]
                w_new = self.patch_size[1]
                w_now = 0
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
