import torch
import json
import os
import pandas as pd

from sidarthe_exp import exp
from torch_euler import Heun, euler, RK4
from populations import populations



if __name__ == '__main__':
    n_epochs = 8000
    runs_directory = "runs_84_ft"

    exp_path = os.path.join(os.getcwd(), "regioni", "sidarthe", "runs_84", "Italy", "05f48624-a181-4753-b736-cd8622d0995d")
    with open(os.path.join(exp_path, "final.json")) as final_settings:
        final_exp_dict = json.load(final_settings)

    with open(os.path.join(exp_path, "settings.json")) as init_settings:
        train_settings = json.load(init_settings)

    params = final_exp_dict["params"]
    learning_rates = train_settings["learning_rates"]
    learning_rates = {key: value * 2 for key, value in learning_rates.items()}
    region = train_settings["region"]
    loss_weights = train_settings["target_weights"]
    train_size = train_settings["train_size"]
    val_len = train_settings["val_len"]
    der_1st_reg = train_settings["der_1st_reg"] * 10
    t_inc = train_settings["t_inc"]
    bound_reg = train_settings["bound_reg"] if "bound_reg" in train_settings else 1e4
    momentum = False
    m = None
    a = None
    integrator = Heun if train_settings == "Heun" else euler
    loss_type = train_settings["loss_type"]


    references = {}
    ref_df = pd.read_csv(os.path.join(os.getcwd(), "regioni/sidarthe_results_new.csv"))
    for key in 'sidarthe':
        references[key] = ref_df[key].tolist()

    for key in ["r0", "h_detected"]:
        references[key] = ref_df[key].tolist()

    for key in params.keys():
        references[key] = ref_df[key].tolist()

    exp(region, populations[region], params,
        learning_rates, n_epochs, region, train_size, val_len,
        loss_weights, der_1st_reg, bound_reg, t_inc, integrator,
        momentum, m, a, loss_type, references, runs_directory)




