"""
Author: Zhou Chen
Date: 2019/12/31
Desc: desc
"""
import numpy as np
from PIL import Image
import os


def tanh(x):
    return np.tanh(x)


def tanh_gradient(x):
    return 1 - tanh(x) ** 2


def relu(x):
    return x * (x > 0)


def relu_gradient(x):
    return 1.0 * (x > 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


def BCE_loss(x, y):
    """
    二分交叉熵损失
    """
    epsilon = 10e-8
    loss = np.sum(-y * np.log(x + epsilon) - (1 - y) * np.log(1 - x + epsilon))
    return loss


def lrelu(x, alpha=0.01):
    return np.maximum(x, x * alpha, x)


def lrelu_gradient(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def img_save(imgs, path, epoch):
    """
    保存每一轮的生成图片
    :param imgs: batch张图片
    :param path:
    :param epoch:
    :return:
    """
    aspect_ratio = 1.0
    border = 1
    border_color = 0
    if not os.path.exists(path):
        os.mkdir(path)
    img_num = imgs.shape[0]

    # 网格
    img_shape = np.array(imgs.shape[1:3])
    img_aspect_ratio = img_shape[1] / float(img_shape[0])
    aspect_ratio *= img_aspect_ratio
    tile_height = int(np.ceil(np.sqrt(img_num * aspect_ratio)))
    tile_width = int(np.ceil(np.sqrt(img_num / aspect_ratio)))
    grid_shape = np.array((tile_height, tile_width))

    tile_img_shape = np.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    tile_img = np.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i * grid_shape[1]
            if img_idx >= img_num:
                break
            img = imgs[img_idx]
            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff + img_shape[0], xoff:xoff + img_shape[1], ...] = img
    file_name = path + "/iteration_{}.png".format(epoch)
    img = Image.fromarray(np.uint8(tile_img * 255), 'L')
    img.save(file_name)
