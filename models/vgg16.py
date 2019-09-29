from keras.layers import (Input,
                          Flatten,
                          Dense,
                          Conv2D,
                          MaxPooling2D)
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

from utils.activations import pitanh
from utils.constants import NB_CHANNELS, NB_ANGLES, NB_SLICE, NB_ESTIMATION_TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH


def build_vgg16_model(config):
    encoder_input = Input(shape=(NB_ESTIMATION_TIMESTEPS, IMAGE_HEIGHT, IMAGE_WIDTH, NB_CHANNELS),
                          name="encoder_input")  # None here is sequence length
    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv1'))(encoder_input)
    x = TimeDistributed(Conv2D(64, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)
    x = TimeDistributed(Flatten())(x)

    estimator_offset_z = TimeDistributed(Dense(NB_SLICE, activation='relu'),
                                         name="estimation_offset_z")(x)
    estimator_rotation = TimeDistributed(Dense(NB_ANGLES, activation=pitanh),
                                         name="estimation_rotation")(x)

    model = Model([encoder_input], outputs=[estimator_offset_z,
                                            estimator_rotation])

    return model
