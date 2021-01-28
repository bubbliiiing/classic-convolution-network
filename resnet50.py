
from __future__ import print_function

import keras.backend as K
import numpy as np
from keras import layers
from keras.applications.imagenet_utils import (decode_predictions,
                                               preprocess_input)
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, Flatten, Input, MaxPooling2D,
                          ZeroPadding2D)
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #-------------------------------#
    #   利用1x1卷积进行通道数的下降
    #-------------------------------#
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用3x3卷积进行特征提取
    #-------------------------------#
    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用1x1卷积进行通道数的上升
    #-------------------------------#
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    #-------------------------------#
    #   利用1x1卷积进行通道数的下降
    #-------------------------------#
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用3x3卷积进行特征提取
    #-------------------------------#
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    #-------------------------------#
    #   利用1x1卷积进行通道数的上升
    #-------------------------------#
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    #-------------------------------#
    #   将残差边也进行调整
    #   才可以进行连接
    #-------------------------------#
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224,224,3],classes=1000):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    # 224,224,3 -> 112,112,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 112,112,64 -> 56,56,64
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 56,56,64 -> 56,56,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 56,56,256 -> 28,28,512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 28,28,512 -> 14,14,1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 14,14,1024 -> 7,7,2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 7,7,2048 -> 1,1,2048
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # 1,1,2048 -> 2048 -> num_classes
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')
    return model


if __name__ == '__main__':
    model = ResNet50()
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models', md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    model.load_weights(weights_path)

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
