import os
import shutil

def organize_images(root_dir):
    """
    将根目录下的PNG图片分别放入以图片名命名的文件夹中
    
    参数:
        root_dir: 根目录路径
    """
    # 获取根目录下所有文件
    files = os.listdir(root_dir)
    
    # 遍历所有文件
    for file in files:
        # 只处理PNG文件
        if file.lower().endswith('.png'):
            # 获取文件名(不含扩展名)
            filename = os.path.splitext(file)[0]
            
            # 构建新文件夹路径
            new_dir = os.path.join(root_dir, filename)
            
            # 如果文件夹不存在则创建
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                print(f'创建文件夹: {new_dir}')
            
            # 构建源文件和目标文件的完整路径
            src_path = os.path.join(root_dir, file)
            dst_path = os.path.join(new_dir, file)
            
            # 移动文件
            shutil.move(src_path, dst_path)
            print(f'移动文件: {file} -> {filename}/{file}')

if __name__ == "__main__":
    # 获取当前脚本所在目录作为根目录
    root_dir = 'D:\\Desktop\\cv\\photo_data_maker\\photo_data_maker-master\\images_marked'
    
    print(f'开始处理目录: {root_dir}')
    organize_images(root_dir)
    print('处理完成!') 