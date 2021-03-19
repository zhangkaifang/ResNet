#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: kaifang zhang
@license: Apache License
@time: 2020/12/01
@contact: 1115291605@qq.com
"""

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, regularizers
from resnet import ResNet18
# from resnet_cbam import ResNet18
from utils import compute_mean_var, lr_schedule_200ep
import numpy as np
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
batchsz = 128

# 1. 归一化函数实现；cifar10 均值和方差，自己计算的。
img_mean = tf.constant([0.50736203482434500, 0.4866895632914611, 0.4410885713465068])
img_std  = tf.constant([0.26748815488001604, 0.2565930997269337, 0.2763085095510783])
def normalize(x, mean=img_mean, std=img_std):
    # x shape: [224, 224, 3]
    # mean：shape为1；这里用到了广播机制。我们安装好右边对齐的原则，可以得到如下；
    # mean : [1, 1, 3], std: [3]        先插入1
    # mean : [224, 224, 3], std: [3]    再变为224
    x = (x - mean)/std
    return x

# 2. 数据预处理，仅仅是类型的转换。    [-1~1]
def preprocess(x, y):
    x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])    # 上下填充4个0，左右填充4个0，变为[40, 40, 3]
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    # x: [0,255]=> 0~1   其次：normalizaion
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1) 调用函数；
    x = normalize(x)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 数据集的加载
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y)            # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
y_test = tf.squeeze(y_test)  # 或者tf.squeeze(y, axis=1)把1维度的squeeze掉。
print(x.shape, y.shape, x_test.shape, y_test.shape)

################## 计算均值和方差 #########################
# x_all = np.concatenate([x, x_test], axis=0).astype(np.float)
# # print(x_all.shape)
# mean_train, std_train = compute_mean_var(x_all/255.)
# print('mean_train:', mean_train, 'std_train:', std_train)
########################################################

# 训练集和标签包装成Dataset对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(5000).map(preprocess).batch(batchsz)
# 测试集和标签包装成Dataset对象
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batchsz)

# 我们来取一个样本，测试一下sample的形状。
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]),
      tf.reduce_max(sample[0]))  # 值范围为[0,1]


def main():
    # 输入：[b, 32, 32, 3]
    model = ResNet18()
    # model = resnet1.ResNet([2, 2, 2], 10)
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    mydense = layers.Dense(100, activation=None)
    fc_net = Sequential([mydense])
    fc_net.build(input_shape=(None, 512))
    fc_net.summary()

    lr = 0.1
    optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=5e-4)
    variables = model.trainable_variables + fc_net.trainable_variables
    for epoch in range(500):

        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]
                out = model(x, training=True)
                avgpool = layers.GlobalAveragePooling2D()(out)
                logits = fc_net(avgpool)
                y_onehot = tf.one_hot(y, depth=100)
                # 多类别交叉熵损失   结果维度[b]
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
                # 添加正则项，所有可以训练的权重添加l2正则项
                loss_regularization = []
                for p in variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
                loss = loss + 5e-4 * loss_regularization

            # 梯度求解
            grads = tape.gradient(loss, variables)
            # 梯度更新
            optimizer.apply_gradients(zip(grads, variables))
            # 学习率动态调整
            lr = lr_schedule_200ep(epoch)
            # 每100个step打印一次
            if step % 100 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss), 'lr:', lr)

        # 做测试
        total_num = 0
        total_correct = 0
        for x, y in test_db:
            out = model(x, training=False)
            avgpool = layers.GlobalAveragePooling2D()(out)
            output = fc_net(avgpool)
            # 预测可能性。
            prob = tf.nn.softmax(output, axis=1)
            pred = tf.argmax(prob, axis=1)  # 还记得吗pred类型为int64,需要转换一下。
            pred = tf.cast(pred, dtype=tf.int32)
            # 拿到预测值pred和真实值比较。
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)  # 转换为numpy数据

        acc = total_correct / total_num
        print('epoch:', epoch, 'test_acc:', acc)


if __name__ == '__main__':
    main()

