import cv2
import numpy as np 
import tensorflow as tf 
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Conv2D, Conv3D,
                                            Dense, Flatten,
                                            GlobalAveragePooling2D,
                                            GlobalMaxPooling2D, Input,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, Dropout, concatenate,
											UpSampling2D)
from tensorflow.python.keras import applications
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K_B
import coloredlogs
from os.path import exists
from input_data import input_data
import os

img_rows = 256
img_cols = 256
checkpt_dir = './train_ckpt'
def create_non_trainable_model(base_model, BOTTLENECK_TENSOR_NAME, use_global_average = False):
    '''
    Parameters
    ----------
    base_model: This is the pre-trained base model with which the non-trainable model is built

    Note: The term non-trainable can be confusing. The non-trainable-parametes are present only in this
    model. The other model (trianable model doesnt have any non-trainable parameters). But if you chose to 
    omit the bottlenecks due to any reason, you will be training this network only. (If you choose
    --omit_bottleneck flag). So please adjust the place in this function where I have intentionally made 
    certain layers non-trainable.

    Returns
    -------
    non_trainable_model: This is the model object which is the modified version of the base_model that has
    been invoked in the beginning. This can have trainable or non trainable parameters. If bottlenecks are
    created, then this network is completely non trainable, (i.e) this network's output is the bottleneck
    and the network created in the trainable is used for training with bottlenecks as input. If bottlenecks
    arent created, then this network is trained. So please use accordingly.
    '''
    # This post-processing of the deep neural network is to avoid memory errors
    x = (base_model.get_layer(BOTTLENECK_TENSOR_NAME).output)
    non_trainable_model = Model(inputs = base_model.input, outputs = [x])
    
    for layer in non_trainable_model.layers:
        layer.trainable = False
    
    return (non_trainable_model)
def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[ :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x

def backend_A(f, weights = None):

    x = Conv2D(512, 3, padding='same', dilation_rate=1, name="dil_A1")(f.output)
    x = Conv2D(512, 3, padding='same', dilation_rate=1, name="dil_A2")(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=1, name="dil_A3")(x)
    x = Conv2D(256, 3, padding='same', dilation_rate=1, name="dil_A4")(x)
    x = Conv2D(128, 3, padding='same', dilation_rate=1, name="dil_A5")(x)
    x = Conv2D(64 , 3, padding='same', dilation_rate=1, name="dil_A6")(x)

    x = Conv2D(1, 1, padding='same', dilation_rate=1, name="dil_A7")(x)
    model = Model(f.input, x, name = "Transfer_learning_model")
    return (model)

def backend_B(f, weights = None):
    
    x = Conv2D(512, 3, padding='same', dilation_rate=2, name="dil_B1")(f.output)
    x = Conv2D(512, 3, padding='same', dilation_rate=2, name="dil_B2")(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=2, name="dil_B3")(x)
    x = Conv2D(256, 3, padding='same', dilation_rate=2, name="dil_B4")(x)
    x = Conv2D(128, 3, padding='same', dilation_rate=2, name="dil_B5")(x)
    x = Conv2D(64 , 3, padding='same', dilation_rate=2, name="dil_B6")(x)
    
    x = Conv2D(1, 1, padding='same', dilation_rate=1,   name="dil_B7")(x)
    model = Model(f.input, x, name = "Transfer_learning_model")
    return (model)

def backend_C(f, weights = None):

    x = Conv2D(512, 3, padding='same', dilation_rate=2, name="dil_C1")(f.output)
    x = Conv2D(512, 3, padding='same', dilation_rate=2, name="dil_C2")(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=2, name="dil_C3")(x)
    x = Conv2D(256, 3, padding='same', dilation_rate=4, name="dil_C4")(x)
    x = Conv2D(128, 3, padding='same', dilation_rate=4, name="dil_C5")(x)
    x = Conv2D(64 , 3, padding='same', dilation_rate=4, name="dil_C6")(x)

    x = Conv2D(1, 1, padding='same', dilation_rate=1, name="dil_C7")(x)
    model = Model(f.input, x, name = "Transfer_learning_model")
    return (model)

def backend_D(f, weights = None):

    x = Conv2D(512, 3, padding='same', dilation_rate=4 , name="dil_D1")(f.output)
    x = Conv2D(512, 3, padding='same', dilation_rate=4 , name="dil_D2")(x)
    x = Conv2D(512, 3, padding='same', dilation_rate=4 , name="dil_D3")(x)
    x = Conv2D(256, 3, padding='same', dilation_rate=4 , name="dil_D4")(x)
    x = Conv2D(128, 3, padding='same', dilation_rate=4 , name="dil_D5")(x)
    x = Conv2D(64 , 3, padding='same', dilation_rate=4 , name="dil_D6")(x)

    x = Conv2D(1, 1, padding='same', dilation_rate=1, name="dil_D7")(x)
    model = Model(f.input, x, name = "Transfer_learning_model")
    return (model)

x = tf.placeholder(tf.float32, shape=[None, img_rows, img_cols, 3], name="input")

base_model = applications.VGG16(input_tensor=x, weights='imagenet', include_top=False, input_shape=(256, 256, 3))
BOTTLENECK_TENSOR_NAME = 'block4_conv3' # This is the 13th layer in VGG16

f = create_non_trainable_model(base_model, BOTTLENECK_TENSOR_NAME) # Frontend

b_a = backend_A(f)

global_step_tensor = tf.train.get_or_create_global_step()

with K_B.get_session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
    if exists(checkpt_dir):
        if tf.train.latest_checkpoint(checkpt_dir) is not None:
            tf.logging.info('Loading Checkpoint from '+ tf.train.latest_checkpoint(checkpt_dir))
            saver.restore(sess, tf.train.latest_checkpoint(checkpt_dir))
    else:
        tf.logging.info('Training from Scratch -  No Checkpoint found')
    
    # iterator = input_data()
    # next_items = iterator.get_next()
		
	

