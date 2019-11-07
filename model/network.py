from keras.backend import image_data_format
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ReLU
from keras.layers import BatchNormalization, Activation, Dense, DepthwiseConv2D


def identity_block(input_tensor, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id='1'):
    
    channel_axis = 1 if image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = input_tensor
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='identity_conv_dw_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='identity_conv_dw_%s_bn' % block_id)(x)
    x = ReLU(name='identity_conv_dw_%s_relu' % block_id)(x)
    x = Conv2D(filters=pointwise_conv_filters, 
               kernel_size=(1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='identity_conv_pw_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='identity_conv_pw_%s_bn' % block_id)(x)
    x = ReLU(name='identity_conv_pw_%s_relu' % block_id)(x)

    return x


def identity_conv_block(input_tensor, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id='1'):

    channel_axis = 1 if image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = input_tensor
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%s_bn' % block_id)(x)
    x = ReLU(name='conv_dw_%s_relu' % block_id)(x)
    x = Conv2D(filters=pointwise_conv_filters,
               kernel_size=(1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%s_bn' % block_id)(x)
    x = ReLU(name='conv_pw_%s_relu' % block_id)(x)

    shortcut = Conv2D(filters=pointwise_conv_filters, kernel_size=(1, 1), strides=strides, padding='same')(input_tensor)
    x = layers.add([x, shortcut])
    return x


def dilation_conv_block(input_tensor, conv_filters, dilation_rate, strides=(1, 1), block_id='1'):
    
    channel_axis = 1 if image_data_format() == 'channels_first' else -1

    x = input_tensor
    if channel_axis == 1:
        input_filters = int(x.shape[1])
    else:
        input_filters = int(x.shape[-1])
    x = Conv2D(filters=input_filters,
               kernel_size=(3, 3),
               padding='valid',
               strides=strides,
               use_bias=False,
               dilation_rate=dilation_rate,   # kernel_extent = dilation_rate * (kernel - 1) + 1
               name='dilation_conv_dw_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='dilation_conv_dw_%s_bn' % block_id)(x)
    x = ReLU(name='dilation_conv_dw_%s_relu' % block_id)(x)
    x = Conv2D(filters=conv_filters, 
               kernel_size=(1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='dilation_conv_pw_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='dilation_conv_pw_%s_bn' % block_id)(x)
    x = ReLU(name='dilation_conv_pw_%s_relu' % block_id)(x)
    
    return x


def module_1(input_tensor, alpha, depth_multiplier, stage):
    x = input_tensor
    x = identity_block(x, 256, alpha, depth_multiplier, strides=(1, 1), block_id=stage+'1')
    x = identity_block(x, 256, alpha, depth_multiplier, strides=(1, 1), block_id=stage+'2')
    x = identity_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=stage+'3')
    return x


def module_2(input_tensor, alpha, depth_multiplier, stage):
    x = input_tensor
    x = identity_block(x, 256, alpha, depth_multiplier, strides=(1, 1), block_id=stage+'1')
    x = identity_block(x, 256, alpha, depth_multiplier, strides=(1, 1), block_id=stage+'2')
    x = identity_conv_block(x, 512, alpha, depth_multiplier, strides=(1, 1), block_id=stage+'3')
    return x


def nn_base(input_tensor=None, alpha=1.0, depth_multiplier=1):                                                 # 3*256*256 input
    
    channel_axis = 1 if image_data_format() == 'channels_first' else -1 
    x = input_tensor
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)                              # 64*128*128
    x = BatchNormalization(axis=channel_axis)(x)
    x = ReLU()(x)
    
    x = identity_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id='1')                     # 64*64*64
    x = identity_block(x, 128, alpha, depth_multiplier, strides=(1, 1), block_id='1')                          # 64*64*64
    x = identity_block(x, 128, alpha, depth_multiplier, strides=(1, 1), block_id='2')                          # 128*64*64
    x_1 = identity_conv_block(x, 256, alpha, depth_multiplier, strides=(1, 1), block_id='2')                   # 256*64*64

    x_2 = module_1(x_1, alpha, depth_multiplier, 'module_1_1')                                                 # 256*32*32
    x_3 = module_1(x_2, alpha, depth_multiplier, 'module_1_2')                                                 # 256*16*16
    x_4 = module_1(x_3, alpha, depth_multiplier, 'module_1_3')                                                 # 256*8*8
    x   = module_1(x_4, alpha, depth_multiplier, 'module_1_4')                                                 # 256*4*4

    x = identity_conv_block(x, 512, alpha, depth_multiplier, strides=(1, 1), block_id='3')                     # 512*4*4
    x = identity_block(x, 512, alpha, depth_multiplier, strides=(1, 1), block_id='3')                          # 512*4*4
    #x_4 = dilation_conv_block(x_4, 256, dilation_rate=2, strides=(1, 1), block_id='2')                        # 512*4*4
    x_4 = identity_conv_block(x_4, 256, alpha, depth_multiplier, strides=(2, 2), block_id='4')                 # 256*4*4
    shortcut_4 = module_2(x_4, alpha, depth_multiplier, 'shortcut_4')                                          # 512*4*4
    x = layers.add([x, shortcut_4])

    x = identity_block(x, 512, alpha, depth_multiplier, strides=(1, 1), block_id='4')                          # 512*4*4
    #x_3 = dilation_conv_block(x_3, 256, dilation_rate=6, strides=(1, 1), block_id='3')                         # 512*4*4
    x_3 = identity_conv_block(x_3, 256, alpha, depth_multiplier, strides=(4, 4), block_id='5')                 # 256*4*4
    shortcut_3 = module_2(x_3, alpha, depth_multiplier, 'shortcut_3')                                          # 512*4*4
    x = layers.add([x, shortcut_3])

    x = identity_block(x, 512, alpha, depth_multiplier, strides=(1, 1), block_id='5')                          # 512*4*4
    #x_2 = dilation_conv_block(x_2, 256, dilation_rate=14, strides=(1, 1), block_id='4')                         # 512*4*4
    x_2 = identity_conv_block(x_2, 256, alpha, depth_multiplier, strides=(8, 8), block_id='6')                 # 256*4*4
    shortcut_2 = module_2(x_2, alpha, depth_multiplier, 'shortcut_2')                                          # 512*4*4
    x = layers.add([x, shortcut_2])

    x = identity_block(x, 512, alpha, depth_multiplier, strides=(1, 1), block_id='6')                          # 512*4*4
    #x_1 = dilation_conv_block(x_1, 256, dilation_rate=30, strides=(1, 1), block_id='5')                         # 512*4*4
    x_1 = identity_conv_block(x_1, 256, alpha, depth_multiplier, strides=(16, 16), block_id='7')               # 256*4*4
    shortcut_1 = module_2(x_1, alpha, depth_multiplier, 'shortcut_1')                                          # 512*4*4
    x = layers.add([x, shortcut_1])
    
    x = identity_block(x, 512, alpha, depth_multiplier, strides=(1, 1), block_id='7')                          # 512*4*4
    #x = identity_block(x, 256, alpha, depth_multiplier, strides=(1, 1), block_id='8')                          # 256*4*4
    #x = identity_block(x, 128, alpha, depth_multiplier, strides=(1, 1), block_id='9')                          # 128*4*4
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax', name = 'predictions')(x)
    
    return x                                                                                





