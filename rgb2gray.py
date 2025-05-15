import os

import cv2
from PIL import Image

##RGB变灰度图
# def batch_convert_to_gray(input_folder_path, output_folder):
#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     # 获取输入文件夹中的所有图像文件
#     for filename in os.listdir(input_folder_path):
#         input_filepath = os.path.join(input_folder_path, filename)
#
#         # 使用 OpenCV 读取图像
#         rgb_image = cv2.imread(input_filepath)
#
#         if rgb_image is not None:
#             # 将 RGB 图像转换为灰度图像
#             gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
#
#             # 构造输出文件路径
#             output_filepath = os.path.join(output_folder_path, f"{filename}")
#
#             # 保存灰度图像
#             cv2.imwrite(output_filepath, gray_image)
#
#             print(f"Converted: {filename}")
#
#
#
# # 示例使用
# input_folder_path = "E:\\BackBone\\image_test\\IR\\"
# output_folder_path = "E:\\BackBone\\image_test\\IR\\"
#
# batch_convert_to_gray(input_folder_path, output_folder_path)

#灰度图变rgb
# import os
# import cv2
#
#
# def batch_convert_to_rgb(input_folder_path, output_folder_path):
#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)
#
#     # 获取输入文件夹中的所有图像文件
#     for filename in os.listdir(input_folder_path):
#         input_filepath = os.path.join(input_folder_path, filename)
#
#         # 使用 OpenCV 读取灰度图像
#         gray_image = cv2.imread(input_filepath, cv2.IMREAD_GRAYSCALE)
#
#         if gray_image is not None:
#             # 将灰度图像转换为 RGB 图像
#             rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
#
#             # 构造输出文件路径
#             output_filepath = os.path.join(output_folder_path, filename)
#
#             # 保存 RGB 图像
#             cv2.imwrite(output_filepath, rgb_image)
#
#             print(f"Converted: {filename}")
#
#
# # 示例使用
# input_folder_path = "E:\BackBone\MSRS\\vi\\"
# output_folder_path = "E:\BackBone\MSRS\\vi1\\"
#
# batch_convert_to_rgb(input_folder_path, output_folder_path)

# 图片重命名
class BatchRename():  #定义一个重命名的类
    def __init__(self):
        #self.path = 'E:\RGB2YCbCr-master\Data\VIF\PET_MRI\MRI\\'
        self.path = '/media/omnisky/SSD2/lmt/RGB2YCbCr-master/Data/VIF/road'


    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 0


        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i).zfill(3) + '.png')  ##重新命名
                file_name = str(i).zfill(3) + '.png'
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()


#修改图片大小
# def resize_images_in_folder(folder_path, target_size=(512, 384)):
#     # 获取文件夹中所有文件名
#     file_names = os.listdir(folder_path)
#
#     # 遍历文件夹中的每个文件
#     for file_name in file_names:
#         # 检查文件是否为图片文件（你可以根据需要调整支持的图片格式）
#         if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#             # 构建完整的文件路径
#             file_path = os.path.join(folder_path, file_name)
#
#             # 打开图像文件
#             img = Image.open(file_path)
#
#             # 调整图像大小
#             resized_img = img.resize(target_size)
#
#             # 保存调整大小后的图像
#             resized_img.save(file_path)
#
#             # 关闭图像文件
#             img.close()
#
# # 调整大小前的图片所在文件夹路径
# input_folder_path = 'E:/BackBone/image_test/IR'
#
# # 调整大小后的图片所在文件夹路径
# output_folder_path = 'E:/BackBone/image_test/IR'
#
# # 调整大小
# resize_images_in_folder(input_folder_path)