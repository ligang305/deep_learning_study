import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pylab


train_dataset = h5py.File("/home/lee/D/data/deep_learning/train_catvnoncat.h5")
test_dataset = h5py.File("/home/lee/D/data/deep_learning/test_catvnoncat.h5")

for key in train_dataset.keys():
    print(key)

# 先用cv读取图片，再用Plt展示图片
# cv_img = cv2.imread("/home/lee/Pictures/Wallpapers/background.png")
# plt.imshow(cv_img)
# plt.show()

# 展示数据集中的图片
img1 = train_dataset['train_set_x'][1]
plt.imshow(img1)
plt.show()

# 展示自己绘制的图
# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()