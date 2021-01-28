
from keras import backend as K
from keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dense,
                          DepthwiseConv2D, GlobalAveragePooling2D, Input,
                          Multiply, Reshape)
from keras.models import Model

#---------------------------------------#
#   激活函数 relu6
#---------------------------------------#
def relu6(x):
    return K.relu(x, max_value=6)

#---------------------------------------#
#   利用relu函数乘上x模拟sigmoid
#---------------------------------------#
def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

#---------------------------------------#
#   用于判断使用哪个激活函数
#---------------------------------------#
def return_activation(x, activation):
    if activation == 'HS':
        x = Activation(hard_swish)(x)
    if activation == 'RE':
        x = Activation(relu6)(x)
    return x

#---------------------------------------#
#   卷积块
#   卷积 + 标准化 + 激活函数
#---------------------------------------#
def conv_block(inputs, filters, kernel, strides, activation):
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization()(x)
    return return_activation(x, activation)

#---------------------------------------#
#   通道注意力机制单元
#   利用两次全连接算出每个通道的比重
#---------------------------------------#
def squeeze(inputs):
    input_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)

    x = Dense(int(input_channels/4))(x)
    x = Activation(relu6)(x)

    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)

    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x

#---------------------------------------#
#   逆瓶颈结构
#---------------------------------------#
def bottleneck(inputs, filters, kernel, up_dim, stride, attention, activation, alpha = 1):
    input_shape = K.int_shape(inputs)
    skip_flag = stride == 1 and input_shape[3] == filters

    #---------------------------------#
    #   part1 利用1x1卷积进行通道上升
    #---------------------------------#
    x = conv_block(inputs, int(up_dim), (1, 1), (1, 1), activation)

    #---------------------------------#
    #   part2 进行3x3的深度可分离卷积
    #---------------------------------#
    x = DepthwiseConv2D(kernel, strides=(stride, stride), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = return_activation(x, activation)

    #---------------------------------#
    #   引入注意力机制
    #---------------------------------#
    if attention:
        x = squeeze(x)

    #------------------------------------------#   
    #   part3 利用1x1卷积进行通道的下降
    #------------------------------------------#
    x = Conv2D(int(alpha * filters), (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if skip_flag:
        x = Add()([x, inputs])

    return x

def MobileNetv3_small(shape = (224,224,3), num_classes = 1000):
    inputs = Input(shape)

    # 224,224,3 -> 112,112,16
    x = conv_block(inputs, 16, (3, 3), strides=(2, 2), activation='HS')

    # 112,112,16 -> 56,56,16
    x = bottleneck(x, 16, (3, 3), up_dim=16, stride=2, attention=True, activation='RE')

    # 56,56,16 -> 28,28,24
    x = bottleneck(x, 24, (3, 3), up_dim=72, stride=2, attention=False, activation='RE')
    x = bottleneck(x, 24, (3, 3), up_dim=88, stride=1, attention=False, activation='RE')
    
    # 28,28,24 -> 14,14,40
    x = bottleneck(x, 40, (5, 5), up_dim=96, stride=2, attention=True, activation='HS')
    x = bottleneck(x, 40, (5, 5), up_dim=240, stride=1, attention=True, activation='HS')
    x = bottleneck(x, 40, (5, 5), up_dim=240, stride=1, attention=True, activation='HS')
    # 14,14,40 -> 14,14,48
    x = bottleneck(x, 48, (5, 5), up_dim=120, stride=1, attention=True, activation='HS')
    x = bottleneck(x, 48, (5, 5), up_dim=144, stride=1, attention=True, activation='HS')

    # 14,14,48 -> 7,7,96
    x = bottleneck(x, 96, (5, 5), up_dim=288, stride=2, attention=True, activation='HS')
    x = bottleneck(x, 96, (5, 5), up_dim=576, stride=1, attention=True, activation='HS')
    x = bottleneck(x, 96, (5, 5), up_dim=576, stride=1, attention=True, activation='HS')

    # 7,7,96 -> 1,1,576
    x = conv_block(x, 576, (1, 1), strides=(1, 1), activation='HS')
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 576))(x)

    # 1,1,576 -> 1,1,1024
    x = Conv2D(1024, (1, 1), padding='same')(x)
    x = return_activation(x, 'HS')

    # 1,1,576 -> 1,1,num_classes
    x = Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(x)
    x = Reshape((num_classes,))(x)

    model = Model(inputs, x)
    return model
    
if __name__ == "__main__":
    model = MobileNetv3_small()
    model.summary()
