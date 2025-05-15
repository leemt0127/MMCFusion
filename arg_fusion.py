import argparse


class args():
    # training args
    epochs = 500  # "number of training epochs, default is 2"
    batch_size = 32  # "batch size for training, default is 4"
    test_batch_size = 1
    # the COCO dataset path in your computer
    # URL: http://images.cocodataset.org/zips/train2014.zip
    vis_dataset = "E:\\three_water\\CMEFusion\\IV_patches\\VIS"
    inf_dataset = "E:\\three_water\\CMEFusion\\IV_patches\\IR"
    HEIGHT = 256
    WIDTH = 256

    test_vis_dataset = "E:\\three_water\\CMEFusion\\IV_patches\\test_image\\VIS"
    test_inf_dataset = "E:\\three_water\\CMEFusion\\IV_patches\\test_image\\IR"

    save_model_dir_autoencoder = "models/CMEFusion_model"
    Final_model_dir = "models"
    save_train_loss_or_not = True
    save_Final_loss_or_not = True
    save_loss_dir = './models/CMEFusion_loss/'

    number = '003'

    cuda = 1
    ssim_weight = [1, 10, 100, 1000, 10000]
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

    lr = 1e-4
    lr_light = 1e-4
    log_interval = 10
    resume = None

    # for test
    # model_default = './checkpoint/03CMEF1.pth'  # test model
    #model_default = './checkpoint/CMEFusion_MSRB2_transformer26/test_net_374.pth'  # test model
    model_default = './checkpoint/test_net_374.pth'  # test model



def train_cfg():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--train_patch_size', type=tuple, default=(256, 256),
                        help='Size of cropped infrared and visible image')
    parser.add_argument('--root', default=r'IV_patches')
    parser.add_argument('--input_nc', default=3)
    parser.add_argument('--output_nc', default=3)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--n_epochs', type=int, default=400,    help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='result/')
    parser.add_argument('--save_folder', type=str, default='checkpoint/')
    parser.add_argument('--test_root', type=str, default=r'IV_patches')
    # parser.add_argument('--test_save_root', type=str, default='DarkFace_output/')
    parser.add_argument('--feature_num', type=int, default=32, help='number of features')
    opt = parser.parse_args()

    return opt


def test_cfg():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='result/')
    parser.add_argument('--train_patch_size', type=tuple)  # default(400,600)
    parser.add_argument('--save_folder', type=str, default='checkpoint/')
    # parser.add_argument('--root', default=r'IV_patches/test')
    parser.add_argument('--root', default=r'image_test')
    opt = parser.parse_args()

    return opt
