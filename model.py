##############################################################################################
# model.py
#
# Dev. Dongwon Paek
# Description: Tensorflow Keras implementation of ReXNet by NAVER
#              Clova AI Corp. in Subclassed Architecture
##############################################################################################

'''
Citation: Dongyoon Han, Sangdoo Yun, Byueongho Heo, Youngjoon Yoo.
          ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network.
          arXiv preprint arXiv:2007.00992, 2020.
'''

import h5py, os, shutil, math

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, DepthwiseConv2D, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense, Input
from tensorflow.keras.layers import ZeroPadding2D, Reshape, Multiply, Add, Concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax, sigmoid

from hyperparameter import *

class swish(Activation):
    def __init__(self):
        super().__init__(Activation)
    def call(self, x):
        return tf.nn.swish(x)

class relu6(Activation):
    def __init__(self):
        super().__init__(Activation)
    def call(self, x):
        return tf.nn.relu6(x)

class relu(Activation):
    def __init__(self):
        super().__init__(Activation)
    def call(self, x):
        return tf.nn.relu(x)

def _add_conv(x, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1, active=True, _relu6=False):
    x.append(Conv2D(filters=channels, kernel_size=(kernel,kernel), strides=stride, padding='same', use_bias=False))
    x.append(BatchNormalization())
    if active:
        x.append(relu6() if _relu6 else relu())

def _add_conv_swish(x, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    x.append(Conv2D(filters=channels, kernel_size=(kernel, kernel), strides=stride, padding='same', use_bias=False))
    x.append(BatchNormalization())
    x.append(swish())

class SE(Layer):
    def __init__(self, in_channels, channels, se_ratio=12):
        super().__init__()
        self.avg_pool = GlobalAveragePooling2D()
        self.conv1 = Conv2D(filters=channels // se_ratio, kernel_size=(1, 1), strides=1, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=channels, kernel_size=(1, 1), strides=1, padding='same')
    
    def call(self, x, training=False):
        y = self.avg_pool(x)
        y = tf.expand_dims(input=y, axis=1)
        y = tf.expand_dims(input=y, axis=1)
        y = self.conv1(y)
        y = self.bn1(y, training=training)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = sigmoid(y)
        return x * y
    
class LinearBottleneck(Layer):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12, **kwargs):
        super().__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels
        
        out = []
        
        if t != 1:
            dw_channels = in_channels * t
            _add_conv_swish(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels
        
        _add_conv(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1, num_group=dw_channels, active=False)
        
        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))
        
        out.append(relu6())
        
        _add_conv(out, in_channels=dw_channels, channels=channels, active=False, _relu6=True)
        
        self.out = Sequential(out)
    
    def call(self, x, training=False):
        out = self.out(x, training=training)
        if self.use_shortcut:
            for i in range(len(out[:, 0:self.in_channels])):
                try:
                    out[:, i] += x[:, i]
                except:
                    print("ERROR!")
            #out[:, 0:self.in_channels] += x
            
        return out

class ReXNetV1(tf.keras.Model):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=4, use_se=True,
                 se_ratio=12, dropout_ratio=0.2, bn_momentum=0.9):
        super().__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        layers = [math.ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        self.depth = sum(layers[:]) * 3

        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        _add_conv_swish(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)

        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        if use_se:
            use_ses = [False] * (layers[0] + layers[1]) + [True] * sum(layers[2:])
        else:
            use_ses = [False] * sum(layers[:])

        for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
            features.append(LinearBottleneck(in_channels=in_c, channels=c, t=t, stride=s, use_se=se, se_ratio=se_ratio))

        pen_channels = int(1280 * width_mult)
        _add_conv_swish(features, c, pen_channels)
        
        features.append(GlobalAveragePooling2D())
        self.features = Sequential(features)
        self.outputs = Sequential([
            Dropout(rate=dropout_ratio),
            Dense(units=classes, activation=softmax, use_bias=True)
        ])
        
    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.outputs(x, training=training)

        return x