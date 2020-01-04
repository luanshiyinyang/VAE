"""
Author: Zhou Chen
Date: 2020/1/2
Desc: desc
"""
from model import VAE
import numpy as np


if __name__ == '__main__':
    x_train = np.load('../data/data.npz')['x'][:5000] / 255.0
    model = VAE()
    model.train(x_train, 100, 0.0001)