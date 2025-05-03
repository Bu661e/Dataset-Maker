import random
import cv2
import numpy as np
from pathlib import Path
import shutil


def n_random(loc, scale, min, max):
    """
    生成指定范围内的随机正态分布数
    :param loc: 平均值
    :param scale: 标准差
    :param min: 范围最小值
    :param max: 范围最大值
    :return: 得到的随机值
    """
    while True:
        n_r = np.random.normal(loc, scale, size=(1, 1))[1][1]
        if min < n_r < max:
            break
    return n_r



def random_flip(img, mode):
    """
    随机翻转图像
    :param img: 输入图像(H,W,C)
    :param mode: 翻转模式
                0: 不翻转(返回原图)
                1: 随机垂直翻转(50%概率)
                2: 随机水平翻转(50%概率)
                3: 随机选择[不翻转, 垂直翻转, 水平翻转, 垂直+水平翻转](25%概率不翻转)
    :return: (翻转后的图像, 翻转标志)
             翻转标志: None(不翻转), 0(垂直翻转), 1(水平翻转), -1(垂直+水平翻转)
    """
    # 定义各模式对应的翻转选项((翻转标志, 返回值))
    flip_options = {
        0: [(None, None)],  # 仅不翻转
        1: [(None, None), (0, 0)],  # 不翻转或垂直翻转
        2: [(None, None), (1, 1)],  # 不翻转或水平翻转
        3: [(None, None), (0, 0), (1, 1), (-1, -1)]  # 四种可能
    }
    
    if mode not in flip_options:
        raise ValueError(f"无效的翻转模式 {mode}，允许值为 0/1/2/3")
    
    # 随机选择一种翻转方式
    flip_flag, ret = random.choice(flip_options[mode])
    
    if flip_flag is not None:
        img = cv2.flip(img, flip_flag)  # 应用翻转
    return img, ret

def random_brightness(
    img: np.ndarray,
    scale_range: tuple[float, float] = (0.7, 1.3),  # 亮度缩放范围 (min_scale, max_scale)，默认±30% (0.7~1.3)
    offset_range: tuple[int, int] = (-50, 30),       # 亮度偏移范围 (min_offset, max_offset)，默认(-50, 30)
    output_range: tuple[int, int] = (2, 254),        # 输出亮度范围 (min, max)，需满足 min ≤ max
    apply_scale: bool = True,                        # 是否应用亮度缩放
    apply_offset: bool = True                        # 是否应用亮度偏移
) -> np.ndarray:
    """
    对图像进行灵活的随机亮度调整（支持缩放和偏移组合）
    
    参数:
        img: 输入图像（H,W,C），numpy数组，数据类型需为uint8或float32（0-255范围）
        scale_range: 亮度缩放范围（乘法因子），格式为(min_scale, max_scale)，其中：
                     min_scale ≤ max_scale，且两者需>0（0表示全黑，无意义）
                     示例：(0.5, 1.5) 表示亮度在50%~150%之间随机变化
        offset_range: 亮度偏移范围（加法因子），格式为(min_offset, max_offset)，其中：
                      min_offset ≤ max_offset，单位为亮度值（0-255对应8位图像）
        output_range: 输出图像的亮度范围截断阈值，格式为(min, max)，用于防止过曝/过暗
                      需满足 0 ≤ min ≤ max ≤ 255
        apply_scale: 是否启用亮度缩放（乘法调整），默认启用
        apply_offset: 是否启用亮度偏移（加法调整），默认启用
    
    返回:
        亮度调整后的图像（与输入图像同尺寸，数据类型为uint8）
    """
    # 参数校验
    # 亮度缩放范围校验
    if not (len(scale_range) == 2 and scale_range[0] <= scale_range[1] and scale_range[0] > 0):
        raise ValueError(f"scale_range需为有效范围(min, max)且min>0，当前值: {scale_range}")
    # 亮度偏移范围校验
    if not (len(offset_range) == 2 and offset_range[0] <= offset_range[1]):
        raise ValueError(f"offset_range需为有效范围(min, max)，当前值: {offset_range}")
    # 输出范围校验
    if not (len(output_range) == 2 and output_range[0] <= output_range[1] and 
            0 <= output_range[0] <= 255 and 0 <= output_range[1] <= 255):
        raise ValueError(f"output_range需为有效亮度范围(0-255)且min≤max，当前值: {output_range}")

    # 生成随机参数
    alpha = random.uniform(*scale_range) if apply_scale else 1.0  # 不缩放时alpha=1
    beta = random.randint(*offset_range) if apply_offset else 0    # 不偏移时beta=0

    # 应用亮度调整
    adjusted = img.astype(np.float32)  # 转换为浮点型以避免溢出
    if apply_scale:
        adjusted *= alpha
    if apply_offset:
        adjusted += beta

    # 截断到输出范围并转换为uint8
    clipped = np.clip(adjusted, output_range[0], output_range[1])
    return np.uint8(clipped)

def random_channel_gain(
    image: np.ndarray,
    gain_range: tuple[float, float] = (0.6, 1.4)  # 通道增益范围 (min_gain, max_gain)，默认±40% (0.6~1.4)
) -> np.ndarray:
    """
    对图像的RGB通道分别应用随机增益，实现色彩平衡调整
    
    参数:
        image: 输入图像（H,W,C），OpenCV格式（BGR通道顺序），数据类型为uint8（0-255）
        gain_range: 通道增益范围，格式为(min_gain, max_gain)，需满足：
                    0 < min_gain ≤ max_gain ≤ 2.0
                    增益=1.0表示无变化，>1增强通道，<1减弱通道
    
    返回:
        通道增益调整后的图像（与输入图像同尺寸，数据类型为uint8，BGR通道顺序）
    
    示例:
        >>> img = cv2.imread("image.jpg")
        >>> adjusted_img = random_channel_gain(img, gain_range=(0.8, 1.2))  # 各通道增益±20%
    """


    # 参数校验
    if not (len(gain_range) == 2 and 0.0 < gain_range[0] <= gain_range[1] <= 2.0):
        raise ValueError(
            f"gain_range需为有效增益范围(0 < min ≤ max ≤ 2)，当前值: {gain_range}"
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"输入图像需为3通道彩色图像，当前形状: {image.shape}")

    # 分离BGR通道
    b_channel, g_channel, r_channel = cv2.split(image)
    print(1)
    
    # 生成各通道随机增益（独立采样）
    gain_b = np.random.uniform(gain_range[0], gain_range[1])
    gain_g = np.random.uniform(gain_range[0], gain_range[1])
    gain_r = np.random.uniform(gain_range[0], gain_range[1])
    
    # 应用增益并截断到0-255
    b_adj = np.clip(b_channel.astype(np.float32) * gain_b, 0, 255).astype(np.uint8)
    g_adj = np.clip(g_channel.astype(np.float32) * gain_g, 0, 255).astype(np.uint8)
    r_adj = np.clip(r_channel.astype(np.float32) * gain_r, 0, 255).astype(np.uint8)
    
    # 合并通道并返回
    return cv2.merge((b_adj, g_adj, r_adj))  # 注意：OpenCV为BGR顺序，直接合并即可



def random_blur(img, r_range):
    """
    随机大小高斯滤波
    :param img: 传入图像
    :param r: 高斯滤波范围，模糊处理边长为2r+1，默认3
    :return: 返回图像
    """
    if r_range[0] < 0 or r_range[1] < 0:
        raise ValueError("r_range 必须为非负整数")
    if r_range[0] > r_range[1]:
        raise ValueError("r_range 必须为 (min, max) 且 min ≤ max")
    
    r = random.randint(r_range[0], r_range[1])
    return cv2.blur(img, (2 * r + 1, 2 * r + 1))


def random_perspective(img, random_range=0.3, symmetry_mode=0, symmetry_direction=0):
    """
    随机透视变换
    :param img:传入图像
    :param random_range:随机范围，默认为0.3，指每个角位置变化距离占边长百分比，应介于0-0.5
    :param symmetry_mode:对称模式，默认为0，即不对称，1为左右对称，2为上下对称。
    以正方形为例，对称将只会把正方形透视变换为等腰梯形，左右对称则左右两边长度相等上下两边平行，上下对称则上下两边长度相等左右两边跑平行。
    :param symmetry_direction:对称方向指定，仅当开启对称情况下有效，默认为0，即不指定。
    若前一参数对称模式为1即左右对称，此参数为1表示等腰梯形上窄下宽，此参数为2反之.
    若前一参数对称模式为2即上下对称，此参数为1表示等腰梯形左宽右窄，此参数为2反之.
    :return:透视后的图像，随机生成透视变换四个点的xy坐标的二维数组
    """
    while True:
        h, w, p = img.shape
        w2 = int(w / 2)
        h2 = int(h / 2)
        if symmetry_mode == 0:
            x1 = random.randint(int(random_range * w), w2)
            x2 = random.randint(int((1 - random_range) * w), w)
            x3 = random.randint(int(random_range * w), w2)
            x4 = random.randint(int((1 - random_range) * w), w)
            y1 = random.randint(int(random_range * h), h2)
            y2 = random.randint(int(random_range * h), h2)
            y3 = random.randint(int((1 - random_range) * h), h)
            y4 = random.randint(int((1 - random_range) * h), h)
            points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        elif symmetry_mode == 1:
            x1 = random.randint(int(random_range * w), w2)
            x2 = w - 1 - x1
            x3 = 0
            x4 = w - 1
            y1 = random.randint(int(random_range * h), h2)
            y2 = y1
            y3 = h - 1
            y4 = y3
            if symmetry_direction == 1:
                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            elif symmetry_direction == 2:
                points = [[x3, y1], [x4, y2], [x1, y3], [x2, y4]]
            elif symmetry_direction == 0:
                if random.choice([1, 2]) == 1:
                    points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                else:
                    points = [[x3, y1], [x4, y2], [x1, y3], [x2, y4]]
            else:
                raise Exception('symmetry_direction只应该是0/1/2')
        elif symmetry_mode == 2:
            x1 = 0
            x2 = random.randint(int((1 - random_range) * w), w)
            x3 = 0
            x4 = x2
            y1 = 0
            y2 = random.randint(int(random_range * h), h2)
            y3 = h - 1
            y4 = h - 1 - y2
            if symmetry_direction == 1:
                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            elif symmetry_direction == 2:
                points = [[x1, y2], [x2, y1], [x3, y4], [x4, y3]]
            elif symmetry_direction == 0:
                if random.choice([1, 2]) == 1:
                    points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                else:
                    points = [[x1, y2], [x2, y1], [x3, y4], [x4, y3]]
            else:
                raise Exception('symmetry_direction只应该是0/1/2')
        else:
            raise Exception('symmetry_mode只应该是0/1/2')

        # 凸四边形验证，不确定修改后是否依然需要，反正留着不会出错
        if ((x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)) * ((x3 - x1) * (y4 - y1) - (y3 - y1) * (x4 - x1)) < 0 and (
                (x1 - x2) * (y3 - y2) - (y1 - y2) * (x3 - x2)) * ((x4 - x2) * (y3 - y2) - (y4 - y2) * (x3 - x2)) < 0:
            break
        else:
            print("warning" + str(points))
    pts3_d1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  # 原图点
    pts3_d2 = np.float32(points)  # 随机得到的四个点
    m = cv2.getPerspectiveTransform(pts3_d1, pts3_d2)  # 矩阵计算
    return cv2.warpPerspective(img, m, (w, h)), points


def points_perspective(img_copy, points):
    """
    进行指定参数的透视变换求纯绿色的最小外接矩形
    :param img_copy: 带绿色标注的图像
    :param points: 透视变换参数点
    :return: 所得矩形的[xc, yc, wc, hc]
    """
    h, w, p = img_copy.shape
    img_copy = cv2.warpPerspective(img_copy, cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [0, h], [w, h]]),
                                                                         np.float32(points)), (w, h))
    cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    img_copy = cv2.inRange(img_copy, np.array([0, 250, 0]), np.array([5, 255, 255]))
    xc, yc, wc, hc = cv2.boundingRect(img_copy)
    # cv2.rectangle(img_copy, (xc, yc), (xc + wc, yc + hc), 255, 2)
    return xc, yc, wc, hc


def overlay(img, back):
    """
    透视变换后的图像叠加到背景图
    :param img: 透视变换后的图像
    :param back: 背景图
    :return: 输出图像，输入图像被放置的位置(左上xy坐标)
    """
    h, w, p = back.shape
    rows, cols, channels = img.shape
    x = random.randint(0, round(w - cols))
    y = random.randint(0, round(h - rows))
    roi = back[y:rows + y, x:cols + x]
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img, img, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    back[y:rows + y, x:cols + x] = dst
    return back, x, y


def delete_file(file_path):
    """
    删除指定文件
    :param file_path: 文件路径，可以是字符串或 Path 对象
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if file_path.exists() and file_path.is_file():
        try:
            file_path.unlink()
            print(f'{file_path} 已成功删除。')
        except OSError as e:
            print(f'删除 {file_path} 时出错: {e}')
    else:
        print(f'{file_path} 不存在或者不是一个文件。')


def delete_empty_folder(folder_path):
    """
    删除指定的空文件夹
    :param folder_path: 文件夹路径，可以是字符串或 Path 对象
    """
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    if folder_path.exists() and folder_path.is_dir() and not any(folder_path.iterdir()):
        try:
            folder_path.rmdir()
            print(f'{folder_path} 已成功删除。')
        except OSError as e:
            print(f'删除 {folder_path} 时出错: {e}')
    else:
        print(f'{folder_path} 不存在、不是一个文件夹或者不是空文件夹。')


def delete_non_empty_folder(folder_path):
    """
    删除指定的非空文件夹
    :param folder_path: 文件夹路径，可以是字符串或 Path 对象
    """
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    if folder_path.exists() and folder_path.is_dir():
        try:
            shutil.rmtree(folder_path)
            print(f'{folder_path} 已成功删除。')
        except OSError as e:
            print(f'删除 {folder_path} 时出错: {e}')
    else:
        print(f'{folder_path} 不存在或者不是一个文件夹。')