from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2 
import random 
random.seed(1337)
import os

data_dir =  "../GaitRecognition/DatasetB_GEI"
img_size = (160,48,1)
# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts
def weight_variable(shape, name):
    initer = tf.truncated_normal_initializer(stddev=0.01)
    W = tf.get_variable(name, dtype=tf.float32, shape=shape, initializer=initer)
    return W
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)


def bias_variable(shape,name):
    b = tf.get_variable(name, dtype=tf.float32, initializer=tf.constant(0.01, shape=shape, dtype=tf.float32))
    return b
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    
def batch_norm_layer(x, train_phase, name=None): #x = [32,14,14,64] =[batch size,h,w, filter數量]
    # training 的時候 拿每個Z(output)批次資料平均值(mu)及標準差(sigma)，所以mu、sigma大小為n*1；beta gamma則為學習的參數，大小也是n*1
    #test 的時候，拿指數加權平均之mu、sigma，以及學習過的beta gamma
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        params_shape = [x.shape[-1]] #batch normorlization 是取最後一個shape 為單位做計算
        print('params_shape',params_shape)
        beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
#         beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = list(range(len(x.shape) - 1)) #[0,1,2]
        print('axises',axises)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
            
        #訓練時用一般的平均及標準差且跑每一次batch 都更新平均及標準差,測試時用moving average之平均及標準差    
#         train_phase = tf.constant(train_phase, dtype = tf.bool)
        mean, var = tf.cond(train_phase, mean_var_with_update, 
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

# def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None): 
#     if norm:
#             # BN for the first input
#             fc_mean, fc_var = tf.nn.moments(xs,axes=[0])
#             scale = tf.Variable(tf.ones([1]))
#             shift = tf.Variable(tf.zeros([1]))
#             epsilon = 0.001
#             # apply moving average for mean and var when train on batch
#             ema = tf.train.ExponentialMovingAverage(decay=0.5)
#             def mean_var_with_update():
#                 ema_apply_op = ema.apply([fc_mean, fc_var])
#                 with tf.control_dependencies([ema_apply_op]):
#                     return tf.identity(fc_mean), tf.identity(fc_var)
#             mean, var = mean_var_with_update()
#             xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)
            
def load_img(path):
    gray_img = cv2.imread(path,0) # shape(160,60)
#         rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB) #shape(160,60,3)
    img = cv2.resize(gray_img,(img_size[1],img_size[0])) 
    img = img.reshape((160,48,1))
    return img


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]
    
def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
#     for d in data:
#         print(d.shape)
    if shuffle:
        data = shuffle_aligned_list(data)
#         print(len(data))

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
#         for d in data:
#             print(d[start:end].shape)
        yield [d[start:end] for d in data]

def batch_generator2(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
#     for d in data:
#         print(d.shape)
    if shuffle:
        data = shuffle_aligned_list(data)
#         print(len(data))

    data_total = len(data[0])
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data_total:
#             print('hhh')
#             for d in data:
#                 print(d[end:data_total].shape)
            yield [d[end:data_total] for d in data]
            batch_count = 0
            
            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
#         for d in data:
#             print(d[start:end].shape)
        yield [d[start:end] for d in data]


def imshow_grid(images, shape=[1, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i], cmap='gray')  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None,model_name=None):
#     color_list = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    num_y = len(set(y))
    
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(d[i]),
#                  color = plt.get_cmap(color_list[d[i]]),
#                  color=plt.cm.Set1(y[i]),
#                  color=plt.cm.bwr(y[i]*100./20),
                 color = y[i],
                 fontdict={'weight': 'bold', 'size': 9})
        
#     vis_x = X[:, 0]
#     vis_y = X[:, 1] 
#     plt.scatter(vis_x, 
#                 vis_y, 
#                 c=y, 
# #                 cmap=plt.cm.get_cmap("jet", num_y)
#                )
#     plt.text(vis_x+0.01, vis_y+0.01, str(d), fontsize=9)      
#     plt.colorbar(ticks=range(num_y))
#     plt.clim(set(y)[0]-0.5, set(y)[-1]-0.5)

    plt.xticks([]), plt.yticks([])
    plt.legend(map(str,(range(62,67))))
    if title is not None:
        plt.title(title+'-'+model_name)
        plt.savefig(model_name+'/'+title+'.jpg')
    plt.show()