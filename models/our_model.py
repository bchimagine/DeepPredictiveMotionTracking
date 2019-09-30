from keras.layers import (Input,
                          Flatten,
                          Dense,
                          CuDNNLSTM, LSTM,
                          Conv2D,
                          MaxPooling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

from utils.activations import (pitanh, PReLU)
from utils.constants import (CHANNEL_AXIS, ROW_AXIS, COL_AXIS, NB_ANGLES, NB_CHANNELS, NB_SLICE,
                             OUR_MODEL_IMAGE_HEIGHT, OUR_MODEL_IMAGE_WIDTH, NB_SINGLE_ANGLE)


def build_our_model(config):

    # ENCODER
    encoder_input = Input(shape=(None, OUR_MODEL_IMAGE_HEIGHT, OUR_MODEL_IMAGE_WIDTH, NB_CHANNELS),
                          name="encoder_input")  # None here is sequence length

    x = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), padding='same',
                               kernel_initializer="he_normal"))(encoder_input)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = PReLU(shared_axes=[ROW_AXIS, COL_AXIS])(x)
    x = TimeDistributed(Conv2D(32, (3, 3), kernel_initializer="he_normal"))(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = PReLU(shared_axes=[ROW_AXIS, COL_AXIS])(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)

    for filter_size in [64, 128, 256, 512]:
        x = TimeDistributed(
            Conv2D(filter_size, (3, 3), padding='same', kernel_initializer="he_normal"))(x)
        x = BatchNormalization(axis=CHANNEL_AXIS)(x)
        x = PReLU(shared_axes=[ROW_AXIS, COL_AXIS])(x)
        x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(x)

    x = TimeDistributed(Flatten())(x)

    encoder_rnn = LSTM(config['network']['hidden_units'], return_sequences=True, return_state=True,
                            name="encoder_rnn")
    encoder_outputs_states = encoder_rnn(x)
    encoder_outputs, encoder_states = encoder_outputs_states[0], encoder_outputs_states[1:]  # [state_h, state_c]

    estimator_offset_z = TimeDistributed(Dense(NB_SLICE, activation='relu'),
                                         name="estimation_offset_z")(encoder_outputs)

    # DECODER
    decoder_input = Input(shape=(None, NB_ANGLES + NB_SLICE), name="decoder_input")
    decoder_rnn = LSTM(config['network']['hidden_units'], return_sequences=True,
                            return_state=True, name="decoder_rnn")
    decoder_outputs_states = decoder_rnn(decoder_input, initial_state=encoder_states)
    decoder_outputs, decoder_states = decoder_outputs_states[0], decoder_outputs_states[1:]  # [state_h, state_c]

    prediction_offset_z = TimeDistributed(Dense(NB_SLICE, activation='relu'),
                                          name="prediction_offset_z")(decoder_outputs)

    if config['network']['single_head']:
        estimator_rotation = TimeDistributed(Dense(NB_ANGLES, activation=pitanh),
                                             name="estimation_rotation")(encoder_outputs)
        prediction_rotation = TimeDistributed(Dense(NB_ANGLES, activation=pitanh),
                                              name="prediction_rotation")(decoder_outputs)
        model = Model([encoder_input, decoder_input], outputs=[estimator_offset_z,
                                                               estimator_rotation,
                                                               prediction_offset_z,
                                                               prediction_rotation,
                                                               ])
    else:
        estimation_rotation_xy = TimeDistributed(Dense(config['network']['hidden_units'], activation='tanh'),
                                                 name="estimation_rotation_xy_fc")(encoder_outputs)
        estimation_rotation_xy = TimeDistributed(Dense(NB_SINGLE_ANGLE + NB_SINGLE_ANGLE, activation=pitanh),
                                                 name="estimation_rotation_xy")(estimation_rotation_xy)
        estimation_rotation_z = TimeDistributed(Dense(config['network']['hidden_units'], activation='tanh'),
                                                name="estimation_rotation_z_fc")(encoder_outputs)
        estimation_rotation_z = TimeDistributed(Dense(NB_SINGLE_ANGLE, activation=pitanh),
                                                name="estimation_rotation_z")(estimation_rotation_z)

        prediction_rotation_xy = TimeDistributed(Dense(config['network']['hidden_units'], activation='tanh'),
                                                 name="prediction_rotation_xy_fc")(decoder_outputs)
        prediction_rotation_xy = TimeDistributed(Dense(NB_SINGLE_ANGLE + NB_SINGLE_ANGLE, activation=pitanh),
                                                 name="prediction_rotation_xy")(prediction_rotation_xy)
        prediction_rotation_z = TimeDistributed(Dense(config['network']['hidden_units'], activation='tanh'),
                                                name="prediction_rotation_z_fc")(decoder_outputs)
        prediction_rotation_z = TimeDistributed(Dense(NB_SINGLE_ANGLE, activation=pitanh),
                                                name="prediction_rotation_z")(prediction_rotation_z)

        model = Model([encoder_input, decoder_input], outputs=[estimator_offset_z,
                                                               estimation_rotation_xy,
                                                               estimation_rotation_z,
                                                               prediction_offset_z,
                                                               prediction_rotation_xy,
                                                               prediction_rotation_z
                                                               ])

    return model

    # PART 2: For variable prediction we create a new model using old weights
    # encoder_predict_model = Model(encoder_input, encoder_states)
    # # encoder_predict_model.summary()

    # decoder_states_inputs = [Input(shape=(config['network']['hidden_units'],)), Input(shape=(config['network']
    # ['hidden_units'],))]
    # decoder_outputs_and_states = decoder_rnn(decoder_input, initial_state=decoder_states_inputs)
    # decoder_outputs_pred, decoder_states_pred = decoder_outputs_and_states[0], decoder_outputs_and_states[1:] # [state_h, state_c]
    # decoder_outputs_pred = predictor(decoder_outputs_pred)
    # decoder_predict_model = Model([decoder_input] + decoder_states_inputs,
    #                               [decoder_outputs_pred] + decoder_states_pred)

    # # decoder_predict_model.summary()

