import multiprocessing
import threading
import yaml
from pathlib import Path
import tarnsform
import cv2
import numpy as np
import random
import time
# import funcs
# import txt_output
from tarnsform import *
from txt_output import *

# 1. cfg文件路径
cfg_DIR = Path(__file__).parent / 'cfg.yaml'
with open(cfg_DIR, 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

# 2. 输入数据和输出数据路径
ROOT = Path(cfg["paths"]['data_root'])
IMG_DIR = ROOT / cfg["paths"]['image']
MARK_DIR = ROOT / cfg["paths"]['mark']
BACK_DIR = ROOT / cfg["paths"]['back']
OUT_DIR = ROOT / cfg["paths"]['output']
print(ROOT)


process_num = multiprocessing.cpu_count() if cfg['process_num'] == -1 else cfg['process_num']
classes = cfg['classes']
format = cfg["output"]['format']

def check():
    # 检查输入图片是否存在
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"未找到图片文件夹{IMG_DIR}")
    if not MARK_DIR.exists():
        raise FileNotFoundError(f"未找到标注文件夹{MARK_DIR}")
    
    # 检查输入图片名是否符合要求
    for img in IMG_DIR.iterdir():
        if img.suffix not in ['.jpg', '.png', '.jpeg']:
            raise ValueError(f"图片文件夹{IMG_DIR}中存在不支持的文件格式: {img.suffix}")
        if img.stem.split('@')[0] not in classes:
            raise ValueError(f"图片文件{img}命名不符合要求，通过@符号分割的前缀必须是类别名")
    for mark in MARK_DIR.iterdir():
        if mark.suffix not in ['.jpg', '.png', '.jpeg']:
            raise ValueError(f"标注文件夹{MARK_DIR}中存在不支持的文件格式: {mark.suffix}")
        if mark.stem.split('@')[0] not in classes:
            raise ValueError(f"标注文件{mark}命名不符合要求，通过@符号分割的前缀必须是类别名")
    
    # 检查输入图片和输入标注是否对应
    if sum([1 for item in IMG_DIR.iterdir()]) != sum([1 for item in MARK_DIR.iterdir()]):
        raise ValueError(f"图片文件夹{IMG_DIR}和标注文件夹{MARK_DIR}中的文件数量不一致")
    marks = [mark.name for mark in MARK_DIR.iterdir()]
    for img in IMG_DIR.iterdir():
        if img.name not in marks:
            raise ValueError(f"图片文件夹{IMG_DIR}和标注文件夹{MARK_DIR}中的文件不对应")
        
    # 检查背景图片是否存在
    if not BACK_DIR.exists():
        raise FileNotFoundError(f"未找到背景图片文件夹{BACK_DIR}")
    if len(list(BACK_DIR.iterdir())) == 0:
        raise ValueError(f"背景图片文件夹{BACK_DIR}中没有背景图片")

def preprocess_folder():
    #  删除上次生成的数据集
    if cfg["output"]["enable_delete_existing"]:
        tarnsform.delete_non_empty_folder(OUT_DIR)

    # 生成output的子文件夹
    # ./output/train
    #                -> images
    #                          -> <class0>
    #                          -> <class1>
    #                -> labels
    #                          --> <class0>
    #                          --> <class1>


    # 1. train set
    # 首先创建 train 目录
    train_dir = OUT_DIR / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    # 创建 train/images 目录
    train_images_dir = train_dir / 'images'
    train_images_dir.mkdir(parents=True, exist_ok=True)

    # 创建 train/labels 目录
    train_labels_dir = train_dir / 'labels'
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    print("/ouput/tarin...已生成")

    # 为每个类别创建子目录
    for cls in classes:
        (train_images_dir / cls).mkdir(parents=True, exist_ok=True)
        (train_labels_dir / cls).mkdir(parents=True, exist_ok=True)

    # 2. image_with_box
    # 如果配置了生成带框的图片，创建带框图片相关目录
    if not cfg["output"]["image_with_box"]["enable"]:
        return
    image_with_box_dir = OUT_DIR / 'image_with_box'
    image_with_box_dir.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        (image_with_box_dir / cls).mkdir(parents=True, exist_ok=True)


    # 3. val set
    # 如果配置了验证集，创建验证集相关目录
    if  not cfg["output"]["validation"]["enable"]:
        return
    val_dir = OUT_DIR / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)

    val_images_dir = val_dir / 'images'
    val_images_dir.mkdir(parents=True, exist_ok=True)

    val_labels_dir = val_dir / 'labels'
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        (val_images_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_labels_dir / cls).mkdir(parents=True, exist_ok=True)
    
    print("/ouput/val...已生成")

def get_cls_index(cls_name):
    for i, cls in enumerate(classes):
        if cls == cls_name:
            return i

def data_marker(img, img_marked, back):
    img_h, img_w = img.shape[:2]
    back_h, back_w = back.shape[:2]

    # 背景图片resize
    if cfg['resize']['enable']:
        back = cv2.resize(back, (cfg['resize']['height'], cfg['resize']['width']))
        back_h, back_w = back.shape[:2]

    # 初始图像翻转
    if cfg['flip']['enable']:
        img, ret = random_flip(img, cfg['flip']['mode'])
        img_marked = cv2.flip(img_marked, ret)
        back, ret = random_flip(back, cfg['flip']['mode'])

    # 亮度随机变化
    img = random_brightness(
        img=img,
        scale_range=cfg["brightness"]["scale"],  
        offset_range=cfg["brightness"]["offset"],       
        output_range=cfg["brightness"]["output_range"],        
        apply_scale=cfg["brightness"]["enable_scale"],                       
        apply_offset=cfg["brightness"]["enable_offset"]   
    )

    back = random_brightness(
        img=back,
        scale_range=cfg["brightness"]["scale"],  
        offset_range=cfg["brightness"]["offset"],       
        output_range=cfg["brightness"]["output_range"],        
        apply_scale=cfg["brightness"]["enable_scale"],                       
        apply_offset=cfg["brightness"]["enable_offset"]   
    )

    # RGB变化
    if cfg['RGB']['enable']:
        img = random_channel_gain(img, cfg['RGB']['range'])
        back = random_channel_gain(back, cfg['RGB']['range'])

    # 缩放
    if cfg['size']['enable']:
        r = random.uniform(cfg['size']['range'][0], cfg['size']['range'][1]) * min(back_h / img_h, back_w / img_w)
        img = cv2.resize(img, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
        img_marked = cv2.resize(img_marked, (0, 0), fx=r, fy=r, interpolation=cv2.INTER_NEAREST)

    # 旋转，报错，在改
    # center = (img_w / 2, img_h / 2)
    # rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=0, scale=1)
    # img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(img_w, img_h))
    # img_marked = cv2.warpAffine(src=img_marked, M=rotate_matrix, dsize=(img_w, img_h))
    

    # 透视变换
    img, points = random_perspective(img, cfg['perspective']['range']
                                     , cfg['perspective']['mode'], cfg['perspective']['direction'])
    # 模糊
    if cfg['blur']['enable']:
        img = random_blur(img, cfg['blur']['range'])
    
    # 寻找标注框
    xc, yc, wc, hc = points_perspective(img_marked, points)
    # 叠加
    back, x, y = overlay(img, back)
    xmin, ymin = (x + xc, y + yc)
    xmax, ymax = (x + xc + wc, y + yc + hc)
    return back, xmin, ymin, xmax, ymax

def maker(process_index, lock):
    train_num = int(cfg['train_num_per_origin_img'] // process_num)
    images = [img for img in IMG_DIR.iterdir()]
    marks = [mark for mark in MARK_DIR.iterdir()]
    backs = [back for back in BACK_DIR.iterdir()]

    if len(images) != len(marks):
        raise ValueError(f"图片文件夹{IMG_DIR}和标注文件夹{MARK_DIR}中的文件数量不一致")
    
    # print(f"process {process_index}: train_num = {train_num}, images: {len(images)}, marks: {len(marks)}, backs: {len(backs)}")
    
    for image, mark in zip(images, marks):
        name = image.stem 
        cls = name.split('@')[0]
  

        # print(f"process {process_index}: image.name: {name}, class: {cls}, class_index = {get_cls_index(cls)}")
        
        image, mark = cv2.imread(str(image)), cv2.imread(str(mark))

        # 训练集
        for i in range(train_num):
            # 注意要使用copy()
            image_copy = image.copy()
            back = cv2.imread(str(random.choice(backs)))
            image_copy, xmin, ymin, xmax, ymax = data_marker(image_copy, mark, back)

            # 1. 写入image
            image_path = OUT_DIR / 'train' / 'images' / cls / f"{name}_{process_index:02d}_{i:05d}.jpg"

            cv2.imwrite(str(image_path), image_copy)

            # 2. 写入label
            label_path = OUT_DIR / 'train' / 'labels' / cls / f"{name}_{process_index:02d}_{i:05d}.txt"
            y, x, n = image_copy.shape
            if format == 'yolo':
                txt = yolo_txt_maker(get_cls_index(cls), xmin, ymin, xmax, ymax, x, y)
            else:
                raise Exception('wrong label_type')
            with open(str(label_path), 'w') as f:
                f.write(txt)

            # 3. 写入image_with_box
            if not cfg["output"]["image_with_box"]["enable"]:
                continue

            if random.random() >= cfg["output"]["image_with_box"]["ratio"]:
                continue

            image_with_box_path = OUT_DIR / 'image_with_box' / cls / f"{name}_{process_index:02d}_{i:05d}.jpg"
            cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), (0, 255, 1), 2)
            cv2.imwrite(str(image_with_box_path), image_copy)


        # 验证集
        if not cfg["output"]["validation"]["enable"]:
            continue

        val_num = int(train_num * cfg["output"]["validation"]["ratio"])
        for i in range(val_num):
            # 注意要使用copy()
            image_copy = image.copy()
            back = cv2.imread(str(random.choice(backs)))
            image_copy, xmin, ymin, xmax, ymax = data_marker(image_copy, mark, back)

            # 1. 写入image
            image_path = OUT_DIR / 'val' / 'images' / cls / f"{name}_{process_index:02d}_{i:05d}.jpg"

            cv2.imwrite(str(image_path), image_copy)

            # 2. 写入label
            label_path = OUT_DIR / 'val' / 'labels' / cls / f"{name}_{process_index:02d}_{i:05d}.txt"
            y, x = image_copy.shape[:2]
            if format == 'yolo':
                txt = yolo_txt_maker(get_cls_index(cls), xmin, ymin, xmax, ymax, x, y)
            else:
                raise Exception('wrong label_type')
            with open(str(label_path), 'w') as f:
                f.write(txt)


def main():
    # 1. 检查image和mark文件夹的内容是否符合要求

    start_time = time.time()
    check()

    # 2. 创建output及其子文件夹
    preprocess_folder()

    # 3. 开启多线程
    thread_lock = threading.Lock()  # 用于线程安全的打印输出
    threads = []
    
    for i in range(process_num):
        # 创建线程，传入线程索引i和锁
        thread = threading.Thread(
            target=maker,
            args=(i, thread_lock)  # 传递锁以便线程内安全打印
        )
        threads.append(thread)
        thread.start()  # 启动线程
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()


    end_time = time.time()
    print("数据种类：", classes)
    print(f"生成{cfg['train_num_per_origin_img'] * len(classes)}张训练集图片，{cfg['train_num_per_origin_img'] * cfg['output']['validation']['ratio'] * len(classes)}张验证集图片")
    print(f"总耗时：{end_time - start_time:.2f}秒")


if __name__ == "__main__":
    main()

