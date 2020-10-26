import torch
from flask import Flask, jsonify
from learning_models.sidarthe import Sidarthe
import os
import json

from learning_models.tied_sidarthe_extended import TiedSidartheExtended

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


@app.route('/api/models/<model_name>/inferences/')
def get_model_inferences(model_name):
    model = TiedSidartheExtended.from_model_summaries(os.path.join(models_root, model_name))
    t_grid = torch.range(0, 365, dtype=torch.float32)
    inferences = model.inference(t_grid)

    return {
        key: value.detach().numpy().tolist() for key, value in inferences.items() if key != "sol"
    }


if __name__ == '__main__':
    app.run()
