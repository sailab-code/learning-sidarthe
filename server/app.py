import torch
from flask import Flask, jsonify
from learning_models.sidarthe import Sidarthe
import os
import json
import numpy as np

from learning_models.tied_sidarthe_extended import TiedSidartheExtended
from dataset.sidarthe_dataset import SidartheDataset

app = Flask(__name__)
models_root = './trained_models'


def get_settings(model_name):
    with open(os.path.join(models_root, model_name, 'settings.json')) as settings_f:
        settings = json.load(settings_f)

    return settings


def get_final_report(model_name):
    with open(os.path.join(models_root, model_name, 'final.json')) as final_f:
        final_report = json.load(final_f)
    return final_report


def get_params(model_name):
    return get_final_report(model_name)['params']


@app.route('/api/')
def hello_world():
    return 'Hello World!'


@app.route('/api/models/')
def get_model_list():
    model_names = [model_dir for model_dir in os.listdir(models_root)
                   if os.path.isdir(os.path.join(models_root, model_dir))]

    model_settings = jsonify([{"model_name":model_name, **get_settings(model_name)} for model_name in model_names])

    return model_settings


@app.route('/api/models/<model_name>')
def get_model_settings(model_name):
    return {"model_name": model_name, **get_settings(model_name)}


@app.route('/api/models/<model_name>/data/')
def get_model_inferences(model_name):
    model_path = os.path.join(models_root, model_name)
    model = TiedSidartheExtended.from_model_summaries(model_path)
    t_grid = torch.range(0, 365, dtype=torch.float32)
    inferences = model.inference(t_grid)

    with open(os.path.join(model_path, 'settings.json')) as settings_f:
        settings = json.load(settings_f)

    dataset = SidartheDataset({"train_size": 52, "val_len": 20, "region": settings['region']})
    dataset.make_dataset()

    return {
        "inferences": {
            key: value.detach().numpy().astype(np.uint32).tolist() for key, value in inferences.items() if key != "sol"
        },
        "targets": {
            key: value.tolist() for key, value in dataset.targets.items()
        }
    }


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5556)
