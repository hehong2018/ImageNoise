#!/usr/bin/env python
# encoding: utf-8
'''
Define some common function
  
Create on 2019-04-25
@author: Hong He
@Change time: 
'''
from __future__ import division
import cv2
import numpy as np
import math


def image_gaussian_blur(kernel_size):
    '''
    2D gaussian noise
    :param sigma:
    :return:
    '''
    imag_h = kernel_size * 2 + 1
    imag_w = kernel_size * 2 + 1
    # center = kernel_size // 2

    gaussian_mat = np.zeros([imag_h, imag_w])
    for i in range(imag_h):
        for j in range(imag_h):
            # print i,j
            gaussian_mat[i][j] = np.exp(-0.5 * (i ** 2 + j ** 2) / (kernel_size ** 2))
    # print gaussian_mat/sum(gaussian_mat)
    return gaussian_mat / sum(gaussian_mat)

    # def gaussian_2d_kernel(kernel_size = 3,sigma = 0):
    #     kernel = np.zeros([kernel_size,kernel_size])
    #     center = kernel_size//2
    #     sum_val = 0
    #     if sigma == 0:
    #         sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8
    #         s = 2*(sigma**2)
    #
    #         for i in range(0,kernel_size):
    #             for j in range(0,kernel_size):
    #                 x = i-center
    #                 y = j-center
    #                 kernel[i,j] = np.exp(-(x**2+y**2) / s)
    #                 sum_val += kernel[i,j]
    #                 #/(np.pi * s)
    #     # sum_val = 1 / sum_val
    #     print kernel / sum_val
    #     return kernel / sum_val


def calc(self, x, y):
    res1 = 1 / (2 * math.pi * self.sigema * self.sigema)
    res2 = math.exp(-(x * x + y * y) / (2 * self.sigema * self.sigema))
    return res1 * res2


def template(radius):
    sideLength = radius * 2 + 1
    result = np.zeros((sideLength, sideLength))
    for i in range(sideLength):
        for j in range(sideLength):
            result[i, j] = calc(i - radius, j - radius)
    all = result.sum()
    return result / all


def RGB(rgb_mat, g_filter, flag=255):
    new_rgb = list()
    for item in rgb_mat:
        new_item = sum(item * g_filter)
        # print new_item
        if new_item > flag:
            new_item = flag
        new_rgb.append(new_item)
    return new_rgb


def calm_RGB(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def imagGaussian(sigma, pixel):
    # noise = np.random.normal(0, sigma, 3)
    noise = image_gaussian_blur(sigma)
    pixel[0] = calm_RGB(pixel[0] + noise[0])
    pixel[1] = calm_RGB(pixel[1] + noise[1])
    pixel[2] = calm_RGB(pixel[2] + noise[2])

    pixel[0] = pixel[0] + noise[0][0]
    pixel[1] = pixel[1] + noise[1][1]
    pixel[2] = pixel[2] + noise[2][2]

    return pixel


def cv_test():
    image = cv2.imread("/home/user/1.png")
    image_change = image.copy()
    data = (100, 300)
    # cv2.imshow("source",cv2.bitwise_and(image,image, mask=cv2.Canny(image, *data)))
    cv2.imshow('img-Canny', cv2.Canny(image, *data))
    cv2.imshow("source", image)
    map_1 = cv2.Canny(image, *data)
    # print map_1
    new_noise_image = cv2.bitwise_and(image, image, mask=cv2.Canny(image, *data))
    sigma = 9
    for i in range(sigma - 1, (image.shape)[0] - sigma + 1):
        for j in range(sigma - 1, (image.shape)[1] - sigma + 1):
            if map_1[i][j] != 0:
                for k in range(3):
                    if np.shape(image_change[i:(i + 2 * sigma + 1), j:(j + 2 * sigma + 1), k]) == (19, 19):
                        tmp = np.multiply(image_gaussian_blur(sigma),
                                          image_change[i:(i + 2 * sigma + 1), j:(j + 2 * sigma + 1), k])
                        # print tmp
                        image_change[i][j][k] = tmp.sum()

    cv2.imshow("gaussian", image_change)

    # cv2.imshow("noise",new_noise_image)



if __name__ == "__main__":
    cv_test()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
