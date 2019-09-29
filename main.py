import os
import argparse
import json
from shutil import copyfile
from datetime import datetime
from keras.optimizers import RMSprop
from keras.models import load_model

from data.data_factory import DataFactory
from models.direct_lstm import build_direct_lstm_model
from models.our_model import build_our_model
from models.resnet18 import build_resnet18_model
from models.vgg16 import build_vgg16_model
from utils.activations import pitanh, PReLU


def get_model(config):
    model_type = config['network']['type'].lower()

    if os.path.exists(config['evaluation']['pretrained_model_path']):
        return load_model(config['evaluation']['pretrained_model_path'],
                          custom_objects={'PReLU': PReLU, 'Pitanh': pitanh})
    else:
        if model_type == 'our_model':
            return build_our_model(config)
        elif model_type == 'direct_lstm':
            return build_direct_lstm_model(config)
        elif model_type == 'resnet18':
            return build_resnet18_model(config)
        elif model_type == 'vgg16':
            return build_vgg16_model(config)


def train_model(model, data_factory):

    val_x, val_y = data_factory.generate_validation_data()

    start_epochs, end_epochs = 0, config["training"]["epoch_steps"]
    for lr in config["training"]["learning_rate"]:
        model.compile(loss='mse', optimizer=RMSprop(lr=lr))
        for epoch in range(start_epochs, end_epochs):
            intra_epochs = config["training"]["intra_epoch_steps"]

            x, y = data_factory.generate_train_data()
            model.fit(x, y,
                      validation_data=(val_x, val_y),
                      batch_size=config["training"]["batch_size"],
                      epochs=intra_epochs * (epoch + 1),
                      initial_epoch=(intra_epochs * epoch))
        start_epochs, end_epochs = end_epochs, end_epochs + config["training"]["epoch_steps"]

    return model


def evaluate_model(model, data_factory):
    test_x, test_y = data_factory.generate_test_data()

    loss = {loss_name: loss for loss_name, loss in zip(model.output_names, model.evaluate(test_x, test_y))}

    print(loss)

    return loss


def export_model_results(model, loss, config_path):
    save_path = os.path.join(config['save_path'], datetime.now().strftime('%Y%m%d%H%M%S'))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        return

    model.save(os.path.join(save_path, config['network']['type']))

    with open(os.path.join(save_path, "results.json"), 'w') as results_file:
        json.dump(loss, results_file)

    copyfile(config_path, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config', type=str, help='path to experiment config file')

    args = parser.parse_args()
    config_path = args.config

    with open(config_path) as f:
        config = json.load(f)

    model = get_model(config)

    # To handle case for evaluation, in case user forgot to specify type of model
    if len(model.output_names) > 4:
        config['network']['single_head'] = False
    else:
        config['network']['single_head'] = True
    data_factory = DataFactory(config)

    if config['evaluation']['pretrained_model_path']:
        model.compile(loss='mse', optimizer='rmsprop')
        loss = evaluate_model(model, data_factory)
    else:
        model = train_model(model, data_factory)

        loss = evaluate_model(model, data_factory)
        export_model_results(model, loss, config_path)

