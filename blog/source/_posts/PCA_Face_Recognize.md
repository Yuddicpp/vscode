---
title: 基于PCA的人脸识别
date: 2020-05-05 12:19:51
tags: 
- PCA
- 人脸识别
categories: Python
cover: /img/PCA_Face_Recognize/top_cover.jpg
top_img: /img/PCA_Face_Recognize/top_cover.jpg
description: PCA（principal components analysis）即主成分分析技术，又称主分量分析。主成分分析也称主分量分析，旨在利用降维的思想，把多指标转化为少数几个综合指标。
---

**PCA旨在利用降维的思想，把多指标转化为少数几个综合指标。**

# 基本概念

​		PCA(Principal Component Analysis)，即主成分分析方法，是一种使用最广泛的数据降维算法。PCA的主要思想是将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，是在原有n维特征的基础上重新构造出来的k维特征。PCA的工作就是从原始的空间中顺序地找一组相互正交的坐标轴，新的坐标轴的选择与数据本身是密切相关的。其中，第一个新坐标轴选择是原始数据中方差最大的方向，第二个新坐标轴选取是与第一个坐标轴正交的平面中使得方差最大的，第三个轴是与第1,2个轴正交的平面中方差最大的。依次类推，可以得到n个这样的坐标轴。通过这种方式获得的新的坐标轴，我们发现，大部分方差都包含在前面k个坐标轴中，后面的坐标轴所含的方差几乎为0。于是，我们可以忽略余下的坐标轴，只保留前面k个含有绝大部分方差的坐标轴。事实上，这相当于只保留包含绝大部分方差的维度特征，而忽略包含方差几乎为0的特征维度，实现对数据特征的降维处理。

# 数学知识

## 协方差与协方差矩阵

### 样本均值

![](/img/PCA_Face_Recognize/mean_value.jpg)

### 样本方差

![](/img/PCA_Face_Recognize/variance.jpg)

### 协方差

![](/img/PCA_Face_Recognize/covariance.jpg)

### 协方差矩阵

![](/img/PCA_Face_Recognize/covariance_matrix.jpg)

## 特征值与特征向量

​	对于给定矩阵A，寻找一个常数λ（可以为复数）和非零向量x，使得向量x被矩阵A作用后所得的向量Ax与原向量x平行，并且满足Ax=λx。其中**常数λ**称为**特征值**，而**非零向量x**称为**特征向量**。

# PCA算法步骤

![](/img/PCA_Face_Recognize/code.jpg)



# 人脸识别代码

```python
#coding:GBK
import os
from numpy import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']

#训练的图片数，每个人总共有十张图片，其他图片用来进行测试
pic_train_num = 9


# 图片矢量化
def img2vector(image):
    img = cv2.imread(image, 0)  # 读取图片
    rows, cols = img.shape  #获取图片的像素
    imgVector = np.zeros((1, rows * cols))
    imgVector = np.reshape(img, (1, rows * cols))#使用imgVector变量作为一个向量存储图片矢量化信息，初始值均设置为0
    return imgVector

#使用的是ORL官方数据集，可以从网址下载到(http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z)
orlpath = "ORL"


# 读入人脸库,每个人随机选择k张作为训练集,其余构成测试集
def load_orl(k):#参数K代表选择K张图片作为训练图片使用
    '''
    对训练数据集进行数组初始化，用0填充，每张图片尺寸都定为112*92,
    现在共有40个人，每个人都选择k张，则整个训练集大小为40*k,112*92
    '''
    train_face = np.zeros((40 * k, 112 * 92))
    train_label = np.zeros(40 * k)  # [0,0,.....0](共40*k个0)
    test_face = np.zeros((40 * (10 - k), 112 * 92))
    test_label = np.zeros(40 * (10 - k))
    # sample=random.sample(range(10),k)#每个人都有的10张照片中，随机选取k张作为训练样本(10个里面随机选取K个成为一个列表)
    sample = random.permutation(10) + 1  # 随机排序1-10 (0-9）+1
    for i in range(40):  # 共有40个人
        people_num = i + 1
        for j in range(10):  # 每个人都有10张照片
            image = orlpath + '/s' + str(people_num) + '/' + str(sample[j]) + '.jpg'
            # 读取图片并进行矢量化
            img = img2vector(image)
            if j < k:
                # 构成训练集
                train_face[i * k + j, :] = img
                train_label[i * k + j] = people_num
            else:
                # 构成测试集
                test_face[i * (10 - k) + (j - k), :] = img
                test_label[i * (10 - k) + (j - k)] = people_num

    return train_face, train_label, test_face, test_label


# 定义PCA算法
def PCA(data, r):#降低到r维
    data = np.float32(np.mat(data))
    rows, cols = np.shape(data)
    data_mean = np.mean(data, 0)  # 对列求平均值
    X = data - np.tile(data_mean, (rows, 1))  # 将所有样例减去对应均值得到A
    C = X * X.T  # 得到协方差矩阵
    D, V = np.linalg.eig(C)  # 求协方差矩阵的特征值和特征向量
    P = V[:, 0:r]  # 按列取前r个特征向量
    P = X.T * P  # 小矩阵特征向量向大矩阵特征向量过渡
    for i in range(r):
        P[:, i] = P[:, i] / np.linalg.norm(P[:, i])  # 特征向量归一化

    Y = X * P
    return Y, data_mean, P

def face_recognize():
    for r in range(10, 41, 10):  # 最多降到40维,即选取前40个主成分（因为当k=1时，只有40维)
        print("当降维到%d时" % (r))
        x_value = []
        y_value = []
        train_face, train_label, test_face, test_label = load_orl(pic_train_num)  # 得到数据集

        # 利用PCA算法进行训练
        data_train_new, data_mean, P = PCA(train_face, r)
        num_train = data_train_new.shape[0]  # 训练脸总数
        num_test = test_face.shape[0]  # 测试脸总数
        temp_face = test_face - np.tile(data_mean, (num_test, 1))
        data_test_new = temp_face * P  # 得到测试脸在特征向量下的数据
        data_test_new = np.array(data_test_new)  # mat change to array
        data_train_new = np.array(data_train_new)

        # 测试准确度
        true_num = 0
        for i in range(num_test):
            testFace = data_test_new[i, :]
            diffMat = data_train_new - np.tile(testFace, (num_train, 1))  # 训练数据与测试脸之间距离
            sqDiffMat = diffMat ** 2
            sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
            sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
            indexMin = sortedDistIndicies[0]  # 距离最近的索引
            if train_label[indexMin] == test_label[i]:
                true_num += 1
            else:
                pass

            accuracy = float(true_num) / num_test
            x_value.append(pic_train_num)
            y_value.append(round(accuracy, 2))

        print('当每个人选择%d张照片进行训练时，The classify accuracy is: %.2f%%' % (pic_train_num, accuracy * 100))

if __name__ == '__main__':
    face_recognize()
```

