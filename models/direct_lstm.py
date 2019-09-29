from keras.layers import (Input,
                          Dense,
                          CuDNNLSTM)
from keras.layers.wrappers import TimeDistributed
from keras.models import Model

from utils.activations import (pitanh)
from utils.constants import (NB_ANGLES, NB_CHANNELS, NB_SLICE,
                             OUR_MODEL_IMAGE_HEIGHT, OUR_MODEL_IMAGE_WIDTH, NB_SINGLE_ANGLE)


def build_direct_lstm_model(config):
    """Returns our model that directly passes image to LSTM

    :param config: dictionary containing value of hyper parameters
    :return: Keras model
    """

    # ENCODER
    encoder_input = Input(shape=(None, OUR_MODEL_IMAGE_HEIGHT, OUR_MODEL_IMAGE_WIDTH, NB_CHANNELS),
                          name="encoder_input")  # None here is sequence length

    encoder_rnn = CuDNNLSTM(config['hidden_units'], return_sequences=True, return_state=True,
                            name="encoder_rnn")
    encoder_outputs_states = encoder_rnn(encoder_input)
    encoder_outputs, encoder_states = encoder_outputs_states[0], encoder_outputs_states[1:]  # [state_h, state_c]

    estimator_offset_z = TimeDistributed(Dense(NB_SLICE, activation='relu'),
                                         name="estimation_offset_z")(encoder_outputs)

    # DECODER
    decoder_input = Input(shape=(None, NB_ANGLES + NB_SLICE), name="decoder_input")
    decoder_rnn = CuDNNLSTM(config['hidden_units'], return_sequences=True, return_state=True, name="decoder_rnn")
    decoder_outputs_states = decoder_rnn(decoder_input, initial_state=encoder_states)
    decoder_outputs, decoder_states = decoder_outputs_states[0], decoder_outputs_states[1:]  # [state_h, state_c]

    prediction_offset_z = TimeDistributed(Dense(NB_SLICE, activation='relu'),
                                          name="prediction_offset_z")(decoder_outputs)

    if config['single_head']:
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
        estimation_rotation_xy = TimeDistributed(Dense(config['hidden_units'], activation='tanh'),
                                                 name="estimation_rotation_xy_fc")(encoder_outputs)
        estimation_rotation_xy = TimeDistributed(Dense(NB_SINGLE_ANGLE + NB_SINGLE_ANGLE, activation=pitanh),
                                                 name="estimation_rotation_xy")(estimation_rotation_xy)
        estimation_rotation_z = TimeDistributed(Dense(config['hidden_units'], activation='tanh'),
                                                name="estimation_rotation_z_fc")(encoder_outputs)
        estimation_rotation_z = TimeDistributed(Dense(NB_SINGLE_ANGLE, activation=pitanh),
                                                name="estimation_rotation_z")(estimation_rotation_z)

        prediction_rotation_xy = TimeDistributed(Dense(config['hidden_units'], activation='tanh'),
                                                 name="prediction_rotation_xy_fc")(decoder_outputs)
        prediction_rotation_xy = TimeDistributed(Dense(NB_SINGLE_ANGLE + NB_SINGLE_ANGLE, activation=pitanh),
                                                 name="prediction_rotation_xy")(prediction_rotation_xy)
        prediction_rotation_z = TimeDistributed(Dense(config['hidden_units'], activation='tanh'),
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

