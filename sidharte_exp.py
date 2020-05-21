import json
import os
from uuid import uuid4

import torch
import numpy as np

from learning_models.sidarthe import Sidarthe
from torch_euler import Heun
from utils.data_utils import select_data
from utils.visualization_utils import generic_plot, Curve, format_xtick, generic_sub_plot, Plot
from torch.utils.tensorboard import SummaryWriter
from populations import populations
from datetime import datetime

def exp(region, population, initial_params, learning_rates, n_epochs, region_name,
        train_size, val_len, loss_weights, der_1st_reg, bound_reg, time_step, integrator,
        momentum, m, a,
        exp_prefix):

    # region directory creation
    # creating folders, if necessary
    base_path = os.path.join(os.getcwd(), "regioni")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    model_name = "sidarthe"
    exp_path = os.path.join(base_path, model_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    # adds directory with the region name
    exp_path = os.path.join(exp_path, region_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    uuid = uuid4()

    # adds directory with the uuid
    exp_path = os.path.join(exp_path, str(uuid))
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    # endregion

    # tensorboard summary
    summary = SummaryWriter(f"runs/{model_name}/{uuid}")

    # creates the json description file with all settings
    description = get_description(region, initial_params, learning_rates, loss_weights, train_size, val_len, der_1st_reg, t_inc, m, a, integrator)
    json_description = json.dumps(description, indent=4)
    json_file = "settings.json"
    with open(os.path.join(exp_path, json_file), "a") as f:
        f.write(json_description)

    # pushes the html version of the summary on tensorboard
    summary.add_text("settings/summary", get_exp_description_html(description, uuid))

    # region target extraction

    # extract targets from csv
    df_file = os.path.join(os.getcwd(), "COVID-19", "dati-regioni", "dpc-covid19-ita-regioni.csv")
    area = [region]
    area_col_name = "denominazione_regione"  # "Country/Region"

    groupby_cols = ["data"]  # ["Date"]

    d_col_name = "isolamento_domiciliare"
    r_col_name = "ricoverati_con_sintomi"
    t_col_name = "terapia_intensiva"
    h_detected_col_name = "dimessi_guariti"
    e_col_name = "deceduti"  # "Fatalities"

    x_target, d_target = select_data(df_file, area, area_col_name, d_col_name, groupby_cols, file_sep=",")
    _, y_target = select_data(df_file, area, area_col_name, "totale_positivi", groupby_cols, file_sep=",")
    _, r_target = select_data(df_file, area, area_col_name, r_col_name, groupby_cols, file_sep=",")
    _, t_target = select_data(df_file, area, area_col_name, t_col_name, groupby_cols, file_sep=",")
    _, h_detected_target = select_data(df_file, area, area_col_name, h_detected_col_name, groupby_cols, file_sep=",")
    _, e_target = select_data(df_file, area, area_col_name, e_col_name, groupby_cols, file_sep=",")

    initial_len = len(y_target)
    tmp_d, tmp_r, tmp_t, tmp_h, tmp_e = [], [], [], [], []
    for i in range(initial_len):
        if y_target[i] > 0:
            tmp_d.append(d_target[i])
            tmp_r.append(r_target[i])
            tmp_t.append(t_target[i])
            tmp_h.append(h_detected_target[i])
            tmp_e.append(e_target[i])
    d_target = tmp_d
    r_target = tmp_r
    t_target = tmp_t
    h_detected_target = tmp_h
    e_target = tmp_e

    targets = {
        "d": d_target,
        "r": r_target,
        "t": t_target,
        "h_detected": h_detected_target,
        "e": e_target
    }

    # endregion

    dataset_size = len(x_target)
    # validation on the next val_len days (or less if we have less data)
    val_size = min(train_size + val_len,
                  len(x_target) - 5)

    params = {
        "alpha": [initial_params["alpha"]] * train_size,
        "beta": [initial_params["beta"]] * train_size,
        "gamma": [initial_params["gamma"]] * train_size,
        "delta": [initial_params["delta"]] * train_size,
        "epsilon": [initial_params["epsilon"]] * train_size,
        "theta": [initial_params["theta"]] * train_size,
        "xi": [initial_params["xi"]] * train_size,
        "eta": [initial_params["eta"]] * train_size,
        "mu": [initial_params["mu"]] * train_size,
        "ni": [initial_params["ni"]] * train_size,
        "tau": [initial_params["tau"]] * train_size,
        "lambda": [initial_params["lambda"]] * train_size,
        "kappa": [initial_params["kappa"]] * train_size,
        "zeta": [initial_params["zeta"]] * train_size,
        "rho": [initial_params["rho"]] * train_size,
        "sigma": [initial_params["sigma"]] * train_size
    }

    model_params = {
        "der_1st_reg": der_1st_reg,
        "population": population,
        "integrator": integrator,
        "time_step": time_step,
        "bound_reg": bound_reg,
        **loss_weights
    }

    train_params = {
        "t_start": 0,
        "t_end": train_size,
        "val_size": val_len,
        "time_step": time_step,
        "m": m,
        "a": a,
        "momentum": True,
        "tensorboard_summary": summary
    }

    sidarthe, logged_info, best_epoch = \
        Sidarthe.train(targets,
                       params,
                       learning_rates,
                       n_epochs,
                       model_params,
                       **train_params)



    summary.flush()



def get_exp_prefix(area, initial_params, learning_rates, train_size, val_len, der_1st_reg,
                   t_inc, m, a, integrator):
    prefix = f"{area[0]}_{integrator.__name__}"
    for key, value in initial_params.items():
        prefix += f"_{key[0]}{value}"

    for key, value in learning_rates.items():
        prefix += f"_{key[0]}{value}"

    prefix += f"_ts{train_size}_vs{val_len}_der1st{der_1st_reg}_tinc{t_inc}_m{m}_a{a}_b{b}"

    prefix += f"{datetime.now().strftime('%B_%d_%Y_%H_%M_%S')}"

    return prefix

def get_description(area, initial_params, learning_rates, target_weights, train_size, val_len, der_1st_reg,
                             t_inc, m, a, integrator):
    return {
        "region": area,
        "initial_values": initial_params,
        "learning_rates": learning_rates,
        "target_weights": target_weights,
        "train_size": train_size,
        "val_len": val_len,
        "der_1st_reg": der_1st_reg,
        "t_inc": t_inc,
        "m": m,
        "a": a,
        "integrator": integrator.__name__,
        "started": datetime.now().strftime('%d/%B/%Y %H:%M:%S')
    }


def get_exp_description_html(description, uuid):
    """
    creates an html representation of the experiment description for tensorboard
    """
    def get_tabs(tabIdx):
        return '&emsp;' * tabIdx

    def get_html_str_from_dict(dictionary, tabIdx=1):
        dict_str = "{<br>"
        for key, value in dictionary.items():
            dict_str += f"{get_tabs(tabIdx)}{key}:"
            if not isinstance(value, dict):
                dict_str += f"{value},<br>"
            else:
                dict_str += get_html_str_from_dict(value, tabIdx + 1) + ",<br>"
        dict_str += get_tabs(tabIdx-1)+"}"
        return dict_str

    description_str = f"Experiment id: {uuid}<br><br>"
    description_str += get_html_str_from_dict(description)

    return description_str


if __name__ == "__main__":
    n_epochs = 2500
    region = "Lombardia"
    params = {
        "alpha": 0.570,
        "beta": 0.0011,
        "gamma": 0.0011,
        "delta": 0.456,
        "epsilon": 0.171,
        "theta": 0.371,
        "xi": 0.125,
        "eta": 0.125,
        "mu": 0.012,
        "ni": 0.027,
        "tau": 0.003,
        "lambda": 0.034,
        "kappa": 0.017,
        "zeta": 0.017,
        "rho": 0.034,
        "sigma": 0.017
    }

    learning_rates = {
        "alpha": 1e-4,
        "beta": 1e-4,
        "gamma": 1e-4,
        "delta": 1e-4,
        "epsilon": 1e-4,
        "theta": 1e-4,
        "xi": 1e-4,
        "eta": 1e-4,
        "mu": 1e-4,
        "ni": 1e-4,
        "tau": 1e-4,
        "lambda": 1e-4,
        "kappa": 1e-4,
        "zeta": 1e-4,
        "rho": 1e-4,
        "sigma": 1e-4
    }

    loss_weights = {
        "d_weight": 1.,
        "r_weight": 1.,
        "t_weight": 1.,
        "h_weight": 1.,
        "e_weight": 1.,
    }

    train_size = 45
    val_len = 20
    der_1st_reg = 1e6
    der_2nd_reg = 0.
    t_inc = 1.

    momentum = True
    m = 0.2
    a = 1.0
    b = 0.05

    bound_reg = 1e6

    integrator = Heun

    exp_prefix = get_exp_prefix(region, params, learning_rates, train_size,
                                val_len, der_1st_reg, t_inc, m, a, integrator)
    print(region)
    exp(region, populations[region], params,
        learning_rates, n_epochs, region, train_size, val_len,
        loss_weights, der_1st_reg, bound_reg, t_inc, integrator,
        momentum, m, a, exp_prefix)
