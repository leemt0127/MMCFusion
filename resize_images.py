import os
from PIL import Image

def resize_images(folder_path, target_size=(256, 256)):
    """
    修改指定文件夹中所有图片的尺寸为目标尺寸。

    Args:
        folder_path (str): 包含图片的文件夹路径。
        target_size (tuple, optional): 目标尺寸 (宽度, 高度)。默认为 (256, 256)。
    """
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹路径 '{folder_path}' 无效。")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                img = Image.open(file_path)
                resized_img = img.resize(target_size)
                resized_img.save(file_path)  # 保存并覆盖原始文件
                print(f"已将图片 '{filename}' 调整为 {target_size}。")
            except Exception as e:
                print(f"处理图片 '{filename}' 时发生错误：{e}")

    print("图片尺寸调整完成。")

if __name__ == "__main__":
    # 预设文件夹路径
    default_folder_path = "/media/omnisky/SSD2/lmt/Evaluation-for-Image-Fusion-main22/Image/Source-Image/MSRS/ir_256"
    use_default = input(f"是否使用默认文件夹路径 '{default_folder_path}'？(yes/no): ").lower()

    if use_default == 'yes':
        folder_path = default_folder_path
    else:
        folder_path = input("请输入要修改图片的文件夹路径：")

    resize_images(folder_path)