import json
import os
from uuid import uuid4

import torch
import numpy as np

from learning_models.sidarthe import Sidarthe
from torch_euler import Heun, euler
from utils.data_utils import select_data
from utils.visualization_utils import generic_plot, Curve, format_xtick, generic_sub_plot, Plot
from torch.utils.tensorboard import SummaryWriter
from populations import populations
from datetime import datetime

def exp(region, population, initial_params, learning_rates, n_epochs, region_name,
        train_size, val_len, loss_weights, der_1st_reg, bound_reg, time_step, integrator,
        momentum, m, a, loss_type,
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
    description = get_description(region, initial_params, learning_rates, loss_weights, train_size, val_len, der_1st_reg, t_inc, m, a, loss_type, integrator)
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
        "population": 1,
        "integrator": integrator,
        "time_step": time_step,
        "bound_reg": bound_reg,
        "loss_type": loss_type,
        **loss_weights
    }

    train_params = {
        "t_start": 0,
        "t_end": train_size,
        "val_size": val_len,
        "time_step": time_step,
        "m": m,
        "a": a,
        "momentum": momentum,
        "tensorboard_summary": summary
    }

    def normalize_values(values, norm):
        """normalize values by a norm, e.g. population"""
        return {key: np.array(value) / norm for key, value in values.items()}

    targets = normalize_values(targets, population)

    sidarthe, logged_info, best_epoch = \
        Sidarthe.train(targets,
                       params,
                       learning_rates,
                       n_epochs,
                       model_params,
                       **train_params)



    with torch.no_grad():
        dataset_size = len(x_target)
        # validation on the next val_len days (or less if we have less data)
        val_size = min(train_size + val_len,
                       len(x_target) - 5)

        t_grid = torch.linspace(0, 100, int(100/t_inc))

        inferences = sidarthe.inference(t_grid)
        inferences = {key: np.array(value) * population for key, value in inferences.items()}

        # region data slices
        t_start = train_params["t_start"]
        train_hat_slice = slice(t_start, int(train_size / t_inc), int(1 / t_inc))
        val_hat_slice = slice(int(train_size / t_inc), int(val_size / t_inc), int(1 / t_inc))
        test_hat_slice = slice(int(val_size / t_inc), int(dataset_size / t_inc), int(1 / t_inc))
        dataset_hat_slice = slice(t_start, int(dataset_size / t_inc), int(1 / t_inc))

        train_target_slice = slice(t_start, train_size, 1)
        val_target_slice = slice(train_size, val_size, 1)
        test_target_slice = slice(val_size, dataset_size, 1)
        dataset_target_slice = slice(t_start, dataset_size, 1)
        # endregion

        # region slice inferences
        def slice_values(values, slice_):
            return {key: value[slice_] for key, value in values.items()}

        hat_train = slice_values(inferences, train_hat_slice)
        hat_val = slice_values(inferences, val_hat_slice)
        hat_test = slice_values(inferences, test_hat_slice)
        hat_dataset = slice_values(inferences, dataset_hat_slice)

        target_train = slice_values(targets, train_target_slice)
        target_val = slice_values(targets, val_target_slice)
        target_test = slice_values(targets, test_hat_slice)
        target_dataset = slice_values(targets, dataset_target_slice)
        # endregion

        # region losses computation
        def extract_losses(losses):
            keys = [
                "rmse",
                "d_rmse",
                "e_rmse",
                "h_rmse",
                "r_rmse",
                "t_rmse"
            ]

            return (losses[key] for key in keys)

        train_risks = sidarthe.losses(
            hat_train,
            target_train
        )

        val_risks = sidarthe.losses(
            hat_val,
            target_val
        )

        test_risks = sidarthe.losses(
            hat_test,
            target_test
        )

        dataset_risks = sidarthe.losses(
            hat_dataset,
            target_dataset
        )
        # endregion

        # region generate final report

        def valid_json_dict(tensor_dict):
            valid_dict = {}
            for key_, value in tensor_dict.items():
                if isinstance(value, torch.Tensor):
                    valid_dict[key_] = value.tolist()
                elif isinstance(value, dict):
                    valid_dict[key_] = valid_json_dict(value)
                else:
                    valid_dict[key_] = value
            return valid_dict

        final_dict = {
            "params": sidarthe.params,
            "best_epoch": best_epoch,
            "train_risks": train_risks,
            "val_risks": val_risks,
            "test_risks": test_risks,
            "dataset_risks": dataset_risks
        }

        json_final = json.dumps(valid_json_dict(final_dict), indent=4)
        json_file = "final.json"
        with open(os.path.join(exp_path, json_file), "a") as f:
            f.write(json_final)

        summary.add_text("settings/final", "Final reporting: " + get_html_str_from_dict(final_dict))
        # endregion

        # region generate plots for final model

        # plot params
        params_plots = sidarthe.plot_params_over_time()
        for (plot, plot_title) in params_plots:
            summary.add_figure(f"final/{plot_title}", plot, close=True, global_step=-1)

        """
        # plot r0
        r0_plot, r0_pl_title = sidarthe.plot_r0(inferences["r0"])
        summary.add_figure(f"fits/{r0_pl_title}", r0_plot, close=True, global_step=epoch)
        """

        # plot inferences

        # get normalized values
        norm_hat_train = normalize_values(hat_train, population)
        norm_hat_val = normalize_values(hat_val, population)
        norm_hat_test = normalize_values(hat_test, population)
        norm_target_train = normalize_values(target_train, population)
        norm_target_val = normalize_values(target_val, population)
        norm_target_test = normalize_values(target_test, population)

        # ranges for train/val/test
        train_range = range(0, train_size)
        val_range = range(train_size, val_size)
        test_range = range(val_size, dataset_size)

        def get_curves(x_range, hat, target, key, color=None):
            pl_x = list(x_range)
            hat_curve = Curve(pl_x, hat, '-', label=f"Estimated {key.upper()}", color=color)
            if target is not None:
                target_curve = Curve(pl_x, target, '.', label=f"Actual {key.upper()}", color=color)
                return [hat_curve, target_curve]
            else:
                return [hat_curve]

        for key in inferences.keys():
            hat_train = norm_hat_train[key]
            hat_val = norm_hat_val[key]
            hat_test = norm_hat_test[key]

            if key in targets:
                # plot inf and target
                target_train = norm_target_train[key]
                target_val = norm_target_val[key]
                target_test = norm_target_test[key]
                pass
            else:
                target_train = None
                target_val = None
                target_test = None
                pass

            train_curves = get_curves(train_range, hat_train, target_train, key, 'r')
            val_curves = get_curves(val_range, hat_val, target_val, key, 'b')
            test_curves = get_curves(test_range, hat_test, target_test, key, 'g')

            tot_curves = train_curves + val_curves + test_curves
            pl_title = f"{key.upper()} - train/validation/test"
            fig = generic_plot(tot_curves, pl_title, None, formatter=format_xtick)
            summary.add_figure(f"final/{key}_global", fig)

        # endregion

    summary.flush()



def get_exp_prefix(area, initial_params, learning_rates, train_size, val_len, der_1st_reg,
                   t_inc, m, a, loss_type, integrator):
    prefix = f"{area[0]}_{integrator.__name__}"
    for key, value in initial_params.items():
        prefix += f"_{key[0]}{value}"

    for key, value in learning_rates.items():
        prefix += f"_{key[0]}{value}"

    prefix += f"_ts{train_size}_vs{val_len}_der1st{der_1st_reg}_tinc{t_inc}_m{m}_a{a}_b{b}_loss{loss_type}"

    prefix += f"{datetime.now().strftime('%B_%d_%Y_%H_%M_%S')}"

    return prefix

def get_description(area, initial_params, learning_rates, target_weights, train_size, val_len, der_1st_reg,
                             t_inc, m, a, loss_type, integrator):
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
        "loss_type": loss_type,
        "started": datetime.now().strftime('%d/%B/%Y %H:%M:%S')
    }


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

def get_exp_description_html(description, uuid):
    """
    creates an html representation of the experiment description for tensorboard
    """



    description_str = f"Experiment id: {uuid}<br><br>"
    description_str += get_html_str_from_dict(description)

    return description_str


if __name__ == "__main__":
    n_epochs = 5000
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
        "alpha": 1e-2,
        "beta": 1e-2,
        "gamma": 1e-2,
        "delta": 1e-2,
        "epsilon": 1e-2,
        "theta": 1e-2,
        "xi": 1e-2,
        "eta": 1e-2,
        "mu": 1e-2,
        "ni": 1e-2,
        "tau": 1e-2,
        "lambda": 1e-2,
        "kappa": 1e-2,
        "zeta": 1e-2,
        "rho": 1e-2,
        "sigma": 1e-2
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
    der_1st_reg = 0.
    der_2nd_reg = 0.
    t_inc = 1.

    momentum = True
    m = 0.1
    a = 1.0
    b = 0.05

    bound_reg = 1e6

    integrator = Heun
    #integrator = euler

    loss_type = "rmse"
    #loss_type = "mape"

    exp_prefix = get_exp_prefix(region, params, learning_rates, train_size,
                                val_len, der_1st_reg, t_inc, m, a, loss_type, integrator)
    print(region)
    exp(region, populations[region], params,
        learning_rates, n_epochs, region, train_size, val_len,
        loss_weights, der_1st_reg, bound_reg, t_inc, integrator,
        momentum, m, a, loss_type, exp_prefix)
