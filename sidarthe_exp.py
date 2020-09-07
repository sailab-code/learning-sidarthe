import json
import os
from uuid import uuid4
import itertools

import torch
import numpy as np

import pandas as pd

from learning_models.sidarthe import Sidarthe
from torch_euler import Heun, euler, RK4
from utils.data_utils import select_data
from utils.visualization_utils import generic_plot, Curve, format_xtick, generic_sub_plot, Plot
from torch.utils.tensorboard import SummaryWriter
from populations import populations
from datetime import datetime

from utils.report_utils import get_exp_prefix, get_description, get_exp_description_html, get_markdown_description


import multiprocessing as mp

verbose = False
normalize = False


def exp(region, population, initial_params, learning_rates, n_epochs, region_name,
        train_size, val_len, loss_weights, der_1st_reg, bound_reg, time_step, integrator,
        momentum, m, a, loss_type, references, runs_directory="runs"):
    # region directory creation
    # creating folders, if necessary
    base_path = os.path.join(os.getcwd(), "regioni")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    model_name = "sidarthe"
    exp_path = os.path.join(base_path, model_name, runs_directory)
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
    summary = SummaryWriter(f"{runs_directory}/{region}/{model_name}/{uuid}")
    exp_prefix = get_exp_prefix(region, initial_params, learning_rates, train_size,
                                val_len, der_1st_reg, time_step, momentum, m, a, loss_type, integrator)

    # creates the json description file with all settings
    description = get_description(region, initial_params, learning_rates, loss_weights, train_size, val_len,
                                  der_1st_reg, time_step, momentum, m, a, loss_type, integrator, bound_reg)
    json_description = json.dumps(description, indent=4)

    json_file = "settings.json"
    with open(os.path.join(exp_path, json_file), "a") as f:
        f.write(json_description)

    # pushes the html version of the summary on tensorboard
    summary.add_text("settings/summary", get_markdown_description(json_description, uuid))

    # region target extraction

    # extract targets from csv

    # if we specify Italy as region, we use national data
    if region == "Italy":
        df_file = os.path.join(os.getcwd(), "COVID-19", "dati-andamento-nazionale", "dpc-covid19-ita-andamento-nazionale.csv")
        area = ["ITA"]
        area_col_name = "stato"  # "Country/Region"
    elif region == "UK":
        df_file = os.path.join(os.getcwd(),"dati-uk", "uk_data_filled.csv")
        area = ["UK"]
        area_col_name = "stato"  # "Country/Region"
    elif region == "FR":
        df_file = os.path.join(os.getcwd(),"dati-fr", "fr_data_processed.csv")
        area = ["FR"]
        area_col_name = "stato"  # "Country/Region"
    else:
        df_file = os.path.join(os.getcwd(), "COVID-19", "dati-regioni", "dpc-covid19-ita-regioni.csv")
        area = [region]
        area_col_name = "denominazione_regione"  # "Country/Region"

    groupby_cols = ["data"]  # ["Date"]

    d_col_name = "isolamento_domiciliare"
    r_col_name = "ricoverati_con_sintomi"
    t_col_name = "terapia_intensiva"
    h_detected_col_name = "dimessi_guariti"
    e_col_name = "deceduti"  # "Fatalities"

    x_target, d_target, dates = select_data(df_file, area, area_col_name, d_col_name, groupby_cols, file_sep=",")
    _, y_target, _ = select_data(df_file, area, area_col_name, "totale_positivi", groupby_cols, file_sep=",")
    _, r_target, _ = select_data(df_file, area, area_col_name, r_col_name, groupby_cols, file_sep=",")
    _, t_target, _ = select_data(df_file, area, area_col_name, t_col_name, groupby_cols, file_sep=",")
    _, h_detected_target, _ = select_data(df_file, area, area_col_name, h_detected_col_name, groupby_cols, file_sep=",")
    _, e_target, _ = select_data(df_file, area, area_col_name, e_col_name, groupby_cols, file_sep=",")

    initial_len = len(y_target)
    tmp_d, tmp_r, tmp_t, tmp_h, tmp_e = [], [], [], [], []
    first_date = None
    for i in range(initial_len):
        if y_target[i] > 0:
            tmp_d = d_target[i:]
            tmp_r = r_target[i:]
            tmp_t = t_target[i:]
            tmp_h = h_detected_target[i:]
            tmp_e = e_target[i:]
            first_date = dates[i]
            break

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



    model_params = {
        "der_1st_reg": der_1st_reg,
        "population": population,
        "integrator": integrator,
        "time_step": time_step,
        "bound_reg": bound_reg,
        "loss_type": loss_type,
        "verbose": verbose,
        "val_size": val_len,
        "train_size": train_size,
        "targets": targets,
        "references": references,
        "first_date": first_date,
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

    if normalize:
        targets = {key: np.array(value) / population for key, value in targets.items()}

    print(f"Starting training for {uuid}")

    sidarthe, logged_info, best_epoch = \
        Sidarthe.train(targets,
                       initial_params,
                       learning_rates,
                       n_epochs,
                       model_params,
                       **train_params)

    with torch.no_grad():
        dataset_size = len(x_target)
        # validation on the next val_len days (or less if we have less data)
        val_size = min(train_size + val_len,
                       len(x_target) - 5)

        t_grid = torch.linspace(0, dataset_size, int(dataset_size / time_step) + 1)

        inferences = sidarthe.inference(t_grid)
        if normalize:
            inferences = {key: np.array(value) * population for key, value in inferences.items()}

        # region data slices
        t_start = train_params["t_start"]
        train_hat_slice = slice(t_start, int(train_size / time_step), int(1 / time_step))
        val_hat_slice = slice(int(train_size / time_step), int(val_size / time_step), int(1 / time_step))
        test_hat_slice = slice(int(val_size / time_step), int(dataset_size / time_step), int(1 / time_step))
        dataset_hat_slice = slice(t_start, int(dataset_size / time_step), int(1 / time_step))

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
        target_test = slice_values(targets, test_target_slice)
        target_dataset = slice_values(targets, dataset_target_slice)

        # endregion

        # region losses computation

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
            "best_epoch": best_epoch,
            "train_risks": train_risks,
            "val_risks": val_risks,
            "test_risks": test_risks,
            "dataset_risks": dataset_risks,
            "params": sidarthe.params
        }

        json_final = json.dumps(valid_json_dict(final_dict), indent=4)
        json_file = "final.json"
        with open(os.path.join(exp_path, json_file), "a") as f:
            f.write(json_final)

        summary.add_text("settings/final", get_markdown_description(json_final, uuid))
        # endregion

        # region generate plots for final model

        # plot params
        params_plots = sidarthe.plot_params_over_time()
        for (plot, plot_title) in params_plots:
            summary.add_figure(f"final/{plot_title}", plot, close=True, global_step=-1)

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
        dataset_range = range(0, dataset_size)

        def get_curves(x_range, hat, target, key, color=None):
            pl_x = list(x_range)
            hat_curve = Curve(pl_x, hat, '-', label=f"Estimated {key.upper()}", color=color)
            if target is not None:
                target_curve = Curve(pl_x, target, '.', label=f"Actual {key.upper()}", color=color)
                return [hat_curve, target_curve]
            else:
                return [hat_curve]

        for key in inferences.keys():

            # skippable keys
            if key in ["sol"]:
                continue

            # separate keys that should be normalized to 1
            if key not in ["r0"]:
                curr_hat_train = norm_hat_train[key]
                curr_hat_val = norm_hat_val[key]
                curr_hat_test = norm_hat_test[key]
            else:
                curr_hat_train = hat_train[key]
                curr_hat_val = hat_val[key]
                curr_hat_test = hat_test[key]


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

            train_curves = get_curves(train_range, curr_hat_train, target_train, key, 'r')
            val_curves = get_curves(val_range, curr_hat_val, target_val, key, 'b')
            test_curves = get_curves(test_range, curr_hat_test, target_test, key, 'g')

            tot_curves = train_curves + val_curves + test_curves

            # get reference in range of interest
            if references is not None:
                ref_y = references[key][dataset_target_slice]
                reference_curve = Curve(dataset_range, ref_y, "--", label="Reference (Nature)")
                tot_curves = tot_curves + [reference_curve]

            pl_title = f"{key.upper()} - train/validation/test/reference"
            fig = generic_plot(tot_curves, pl_title, None, formatter=sidarthe.format_xtick)
            summary.add_figure(f"final/{key}_global", fig)

        # endregion

    summary.flush()
    return sidarthe, uuid, val_risks[sidarthe.val_loss_checked]

if __name__ == "__main__":
    n_epochs = 8000
    region = "Italy"
    runs_directory = "runs_110_test_chi"
    params = {
        "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * 64,
        "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 63),
        "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 10 + [0.11] * 64,
        "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 63),
        "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * 64,
        "theta": [0.371],
        "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * 64,
        "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * 64,
        "mu": [0.017] * 22 + [0.008] * (17 + 63),
        "nu": [0.027] * 22 + [0.015] * (17 + 63),
        "tau": [0.15] * 102,
        "lambda": [0.034] * 22 + [0.08] * (17 + 63),
        "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * 64,
        "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * 64,
        "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * 64,
        "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * 64,
        "phi": [0.02] * 102,
        "chi": [0.02] * 102
    }

    sizes = {
        key: len(value) for key, value in params.items()
    }

    print(sizes)

    learning_rates = {
        "alpha": 1e-5,
        "beta": 1e-6,
        "gamma": 1e-5,
        "delta": 1e-6,
        "epsilon": 1e-5,
        "theta": 1e-7,
        "xi": 1e-5,
        "eta": 1e-5,
        "mu": 1e-5,
        "nu": 1e-5,
        "tau": 1e-5,
        "lambda": 1e-5,
        "kappa": 1e-5,
        "zeta": 1e-5,
        "rho": 1e-5,
        "sigma": 1e-5,
        "phi": 1e-5,
        "chi": 1e-5
    }

    loss_weights = {
        "d_weight": 1.,
        "r_weight": 1.,
        "t_weight": 1.,
        "h_weight": 1.,
        "e_weight": 1.,
    }

    train_size = 110
    val_len = 5
    der_1st_reg = 50000.0
    der_2nd_reg = 0.
    t_inc = 1.

    momentum = True
    m = 0.125
    a = 0.1
    bound_reg = 1e5
    integrator = Heun
    loss_type = "rmse"
    print(region)

    references = {}
    ref_df = pd.read_csv(os.path.join(os.getcwd(), "regioni/sidarthe_results_new.csv"))
    for key in 'sidarthe':
        references[key] = ref_df[key].tolist()

    for key in ["r0", "h_detected"]:
        references[key] = ref_df[key].tolist()

    for key in params.keys():
        if key in ref_df:
            references[key] = ref_df[key].tolist()

    exp(region, populations[region], params,
                            learning_rates, n_epochs, region, train_size, val_len,
                            loss_weights, der_1st_reg, bound_reg, t_inc, integrator,
                            momentum, m, a, loss_type, references, runs_directory)