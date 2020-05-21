import os
from time import sleep

import pylab as pl
# import matplotlib.pyplot as pl
import torch
import datetime
import numpy as np

from learning_models.new_torch_sir import NewSir
from learning_models.torch_sir import SirEq
from torch_euler import Heun, euler
from utils.data_utils import select_data
from utils.visualization_utils import generic_plot, Curve, format_xtick, generic_sub_plot, Plot
from torch.utils.tensorboard import SummaryWriter
from populations import populations
from datetime import datetime


def exp(region, population, beta_t0, gamma_t0, delta_t0, lr_b, lr_g, lr_d, lr_a, n_epochs, name, train_size, val_len,
        der_1st_reg, der_2nd_reg, use_alpha, y_loss_weight, t_inc, exp_prefix, integrator, m, a, b):
    df_file = os.path.join(os.getcwd(), "COVID-19", "dati-regioni", "dpc-covid19-ita-regioni.csv")
    # df_file = os.path.join(os.getcwd(), "train.csv")
    area = [region]  # list(df["denominazione_regione"].unique())
    area_col_name = "denominazione_regione"  # "Country/Region"
    value_col_name = "deceduti"  # "Fatalities"
    groupby_cols = ["data"]  # ["Date"]

    x_target, w_target = select_data(df_file, area, area_col_name, value_col_name, groupby_cols, file_sep=",")
    _, y_target = select_data(df_file, area, area_col_name, "totale_positivi", groupby_cols, file_sep=",")
    _, healed = select_data(df_file, area, area_col_name, "dimessi_guariti", groupby_cols, file_sep=",")

    initial_len = len(y_target)
    tmp_y, tmp_w, tmp_h = [], [], []
    for i in range(len(y_target)):
        if y_target[i] > 0:
            tmp_y.append(y_target[i])
            tmp_w.append(w_target[i])
            tmp_h.append(healed[i])
    y_target = tmp_y
    w_target = tmp_w
    healed = tmp_h

    # creating folders, if necessary
    base_path = os.path.join(os.getcwd(), "regioni")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    exp_path = os.path.join(base_path, "new_sir_test")
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    exp_path = os.path.join(exp_path, name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    beta = [beta_t0 for _ in range(int(train_size))]
    gamma = [gamma_t0]
    delta = [delta_t0]
    summary = SummaryWriter(f"runs/{name}/{exp_prefix}")

    targets = {
        "w": w_target,
        "y": y_target
    }

    initial_params = {
        "beta": beta,
        "gamma": gamma,
        "delta": delta
    }

    learning_rates = {
        "beta": lr_b,
        "gamma": lr_g,
        "delta": lr_d
    }

    model_params = {
        "der_1st_reg": der_1st_reg,
        "der_2nd_reg": der_2nd_reg,
        "population": population,
        "y_loss_weight": y_loss_weight,
        "integrator": integrator,
        "t_inc": t_inc
    }

    train_params = {
        "t_start": 0,
        "t_end": train_size,
        "val_size": val_len,
        "t_inc": t_inc,
        "m": m,
        "a": a,
        "b": b,
        "momentum": True,
        "tensorboard_summary": summary
    }

    sir, logged_info, best_epoch = \
        NewSir.train(targets,
                     initial_params,
                     learning_rates,
                     n_epochs,
                     model_params,
                     **train_params)

    with torch.no_grad():
        val_size = min(train_size + val_len,
                       len(w_target) - 5)  # validation on the next val_len days (or less if we have less data)
        dataset_size = len(w_target)

        inferences = sir.inference(torch.linspace(0, 100, int(100 / t_inc)))
        w_hat = inferences["w"]
        y_hat = inferences["y"]
        sol = inferences["sol"]

        t_start = train_params["t_start"]
        train_hat_slice = slice(t_start, int(train_size / t_inc), int(1 / t_inc))
        val_hat_slice = slice(int(train_size / t_inc), int(val_size / t_inc), int(1 / t_inc))
        test_hat_slice = slice(int(val_size / t_inc), int(dataset_size / t_inc), int(1 / t_inc))
        dataset_hat_slice = slice(t_start, int(dataset_size / t_inc), int(1 / t_inc))

        train_target_slice = slice(t_start, train_size, 1)
        val_target_slice = slice(train_size, val_size, 1)
        test_target_slice = slice(val_size, dataset_size, 1)
        dataset_target_slice = slice(t_start, dataset_size, 1)

        def slice_values(values, slice):
            return {key: value[slice] for key, value in values.items()}

        hat_train = slice_values(inferences, train_hat_slice)
        hat_val = slice_values(inferences, val_hat_slice)
        hat_test = slice_values(inferences, test_hat_slice)
        hat_dataset = slice_values(inferences, dataset_hat_slice)

        target_train = slice_values(targets, train_target_slice)
        target_val = slice_values(targets, val_target_slice)
        target_test = slice_values(targets, test_hat_slice)
        target_dataset = slice_values(targets, dataset_target_slice)

        def extract_losses(losses):
            return losses["mse"], losses["w_mse"], losses["y_mse"]

        train_risk, train_w_risk, train_y_risk = extract_losses(
            sir.losses(
                hat_train,
                target_train
            )
        )

        validation_risk, validation_w_risk, validation_y_risk = extract_losses(
            sir.losses(
                hat_val,
                target_val
            )
        )

        test_risk, test_w_risk, test_y_risk = extract_losses(
            sir.losses(
                hat_test,
                target_test
            )
        )

        dataset_risk, _, _ = extract_losses(
            sir.losses(
                hat_dataset,
                target_dataset
            )
        )

        mse_losses = [info["mse"].detach().numpy() for info in logged_info]

        log_file = os.path.join(exp_path, exp_prefix + "sir_" + area[0] + "_results.txt")
        with open(log_file, "w") as f:
            f.write("Beta:\n ")
            f.write(str(list(sir.beta.detach().numpy())) + "\n")
            f.write("Gamma:\n ")
            f.write(str(list(sir.gamma.detach().numpy())) + "\n")
            f.write("Delta:\n ")
            f.write(str(list(sir.delta.detach().numpy())) + "\n")
            f.write("Train Risk:\n")
            f.write(str(train_risk.detach().numpy()) + "\n")
            f.write("Validation Risk:\n")
            f.write(str(validation_risk.detach().numpy()) + "\n")
            f.write("Test Risk:\n")
            f.write(str(test_risk.detach().numpy()) + "\n")
            f.write("Dataset Risk:\n")
            f.write(str(dataset_risk.detach().numpy()) + "\n")
            f.write("Loss over epochs: \n")
            f.write(str(mse_losses) + "\n")

        csv_file = os.path.join(exp_path, "scores.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, "w") as f:
                f.write("name\tbeta_t0\tgamma_t0\tdelta_t0\tbeta\tgamma\tdelta\tlr_beta\tlr_gamma\tlr_delta\t"
                        "train_size\tval_size\tfirst_derivative_reg\tsecond_derivative_reg\tuse_alpha\ty_loss_weight\tt_inc\t"
                        "w_train_risk\tw_val_risk\tw_test_risk\t"
                        "train_risk\tval_risk\ttest_risk\tdataset_risk\tbest_epoch\tintegrator\tm\ta\tb\n")

        with open(csv_file, "a") as f:
            _res_str = '\t'.join(
                [exp_prefix, str(beta_t0), str(gamma_t0), str(delta_t0),
                 str(list(sir.beta.detach().numpy())).replace("\n", " "),
                 str(list(sir.gamma.detach().numpy())).replace("\n", " "),
                 str(list(sir.delta.detach().numpy())).replace("\n", " "),
                 str(learning_rates["beta"]), str(learning_rates["gamma"]), str(learning_rates["delta"]),
                 str(train_size), str(val_len),
                 str(der_1st_reg), str(der_2nd_reg), str(use_alpha), str(y_loss_weight), str(t_inc),
                 str(train_w_risk.detach().numpy()), str(validation_w_risk.detach().numpy()),
                 str(test_w_risk.detach().numpy()),
                 str(train_risk.detach().numpy()), str(validation_risk.detach().numpy()),
                 str(test_risk.detach().numpy()), str(dataset_risk.detach().numpy()),
                 str(best_epoch), integrator.__name__, str(m), str(a), str(b)
                 ]) + '\n'
            f.write(_res_str)

        # Plotting
        file_format = ".png"

        # ------------------------------------ #
        # BETA, GAMMA, DELTA plot
        bgd_pl_title = "$\\beta, \\gamma, \\delta$  ({}".format(str(area[0])) + str(")")
        bgd_pl_path = os.path.join(exp_path, exp_prefix + "_bcd_over_time" + file_format)

        pl_x = list(range(train_size))  # list(range(len(beta)))
        beta_pl = Curve(pl_x, sir.beta.detach().numpy(), '-g', "$\\beta$")
        gamma_pl = Curve(pl_x, [sir.gamma.detach().numpy()] * train_size, '-r', "$\\gamma$")
        delta_pl = Curve(pl_x, [sir.delta.detach().numpy()] * train_size, '-b', "$\\delta$")
        params_curves = [beta_pl, gamma_pl, delta_pl]
        fig = generic_plot(params_curves, bgd_pl_title, bgd_pl_path, formatter=format_xtick)
        summary.add_figure("final/params_over_time", figure=fig)

        # R0 Plot
        r0_pl_title = '$R_0$  ({}'.format(str(area[0])) + str(")")
        r0_pl_path = os.path.join(exp_path, exp_prefix + "_r0" + file_format)

        pl_x = list(range(len(beta)))
        r0_pl = Curve(pl_x, sir.beta.detach().numpy() / sir.gamma.detach().numpy(), '-', label="$R_0$")
        thresh_r0_pl = Curve(pl_x, [1.0] * len(pl_x), '-', color="magenta")
        fig = generic_plot([r0_pl, thresh_r0_pl], r0_pl_title, r0_pl_path, formatter=format_xtick)
        summary.add_figure("final/r0", figure=fig)

        # normalized values
        def normalize_values(values, norm):
            return {key: np.array(value) / norm for key, value in values.items()}

        w_hat = hat_dataset["w"].detach().numpy() / population
        sol = inferences["sol"].detach().numpy() / population
        norm_targets = normalize_values(targets, population)
        norm_hat_train = normalize_values(hat_train, population)
        norm_hat_val = normalize_values(hat_val, population)
        norm_hat_test = normalize_values(hat_test, population)
        norm_hat_dataset = normalize_values(hat_dataset, population)
        norm_target_train = normalize_values(target_train, population)
        norm_target_val = normalize_values(target_val, population)
        norm_target_test = normalize_values(target_test, population)
        norm_target_dataset = normalize_values(target_dataset, population)

        norm_healed = np.array(healed) / population
        norm_recovered = norm_healed + norm_targets["w"]

        # ------------------------------------ #
        # SIR dynamic plot
        sir_global_path = os.path.join(exp_path, exp_prefix + "_SIR_global" + file_format)
        sir_title = 'SIR  ({}'.format(region) + str(")")

        # x grid for inferred values
        sir_len = len(sol[:,0])
        pl_sir_x = np.arange(0, sir_len, t_inc)

        # x grid for target values
        w_target_len = len(norm_targets["w"])
        pl_target_x = np.arange(0, w_target_len, t_inc)

        # susceptible subplot
        s_fit_curve = Curve(pl_sir_x, sol[:, 0], '-g', label='$x$')

        s_truth = np.ones(w_target_len) - (norm_targets["y"] + norm_recovered)
        s_truth_curve = Curve(pl_target_x, s_truth, '.g', label='$x$')
        s_curves = [s_fit_curve, s_truth_curve]

        s_subplot = Plot(x_label=None, y_label="S", use_grid=True, use_legend=True,
                         curves=s_curves,
                         bottom_adjust=0.15, margins=0.05, formatter=format_xtick,
                         h_pos=1, v_pos=1)

        # infectious subplot
        i_fit_curve = Curve(pl_sir_x, sol[:, 1], '-r', label='$y$')
        i_truth_curve = Curve(pl_target_x, norm_targets["y"], '.r', label='$y$')
        i_curves = [i_fit_curve, i_truth_curve]
        i_subplot = Plot(x_label=None, y_label="I", use_grid=True, use_legend=True,
                         curves=i_curves,
                         bottom_adjust=0.15,
                         margins=0.05, formatter=format_xtick,
                         h_pos=1, v_pos=2)

        # recovered subplot
        r_fit_curve = Curve(pl_sir_x, sol[:, 2], '-k', label='$z$')
        r_truth_points = Curve(pl_target_x, norm_recovered, '.k', label='$z$')
        r_curves = [r_fit_curve, r_truth_points]
        r_subplot = Plot(x_label=None, y_label="R", use_grid=True, use_legend=True,
                         curves=r_curves,
                         bottom_adjust=0.15,
                         margins=0.05, formatter=format_xtick,
                         h_pos=1, v_pos=3)

        fig = generic_sub_plot([s_subplot, i_subplot, r_subplot], sir_title, sir_global_path)
        summary.add_figure("final/sir_dynamic", figure=fig)

        # ------------------------------------ #
        # deaths plot
        deaths_pl_title = 'Deaths  ({}'.format(str(area[0])) + str(")")
        deaths_pl_path = os.path.join(exp_path, exp_prefix + "_W_global" + file_format)

        w_fit_curve = Curve(range(0, w_hat.shape[0]), w_hat, '-', label='$w$')
        fig = generic_plot([w_fit_curve], deaths_pl_title, deaths_pl_path, 'Time in days', 'Deaths')
        summary.add_figure("final/deaths_predicted", figure=fig)

        # ------------------------------------ #
        # ranges for train/val/test
        train_range = range(0, train_size)
        val_range = range(train_size, val_size)
        test_range = range(val_size, dataset_size)

        # comparison of infectious train/val/test
        infected_pl_title = 'Infectious  ({}'.format(str(area[0])) + str(")")
        infected_pl_path = os.path.join(exp_path, exp_prefix + "_I_fit" + file_format)

        y_hat_train = Curve(train_range, norm_hat_train["y"], "-r", color="red",
                            label='$y$ fit')
        y_hat_val = Curve(list(range(train_size, val_size)), norm_hat_val["y"], '-r',
                          color='darkblue', label='$y$ validation')
        y_hat_test = Curve(list(range(val_size, dataset_size)), norm_hat_test["y"], '-r', color='orange',
                           label='$y$ prediction')

        y_truth_train = Curve(train_range, norm_target_train["y"], '.r', color="red",
                              label='$\\hat{y}$ fit', )
        y_truth_val = Curve(val_range, norm_target_val["y"], '.', color="darkblue",
                            label='$\\hat{y}$ validation')
        y_truth_test = Curve(test_range, norm_target_test["y"], '.', color="orange",
                             label='$\\hat{y}$ prediction')

        fig = generic_plot([y_hat_train, y_hat_val, y_hat_test, y_truth_train, y_truth_val, y_truth_test],
                           infected_pl_title, infected_pl_path, y_label='Infectious', formatter=format_xtick)
        summary.add_figure("final/infected_fit", figure=fig)

        # comparison of deaths train/val/test
        w_pl_title = 'Deaths  ({}'.format(str(area[0])) + str(")")
        w_pl_path = os.path.join(exp_path, exp_prefix + "_W_fit" + file_format)

        w_hat_train = Curve(train_range, norm_hat_train["w"], "-r", color="red",
                            label='$w$ fit')
        w_hat_val = Curve(list(range(train_size, val_size)), norm_hat_val["w"], '-r',
                          color='darkblue', label='$w$ validation')
        w_hat_test = Curve(list(range(val_size, dataset_size)), norm_hat_test["w"], '-r', color='orange',
                           label='$w$ prediction')

        w_truth_train = Curve(train_range, norm_target_train["w"], '.r', color="red",
                              label='$\\hat{w}$ fit')
        w_truth_val = Curve(val_range, norm_target_val["w"], '.', color="darkblue",
                            label='$\\hat{w}$ validation')
        w_truth_test = Curve(test_range, norm_target_test["w"], '.', color="orange",
                             label='$\\hat{w}$ prediction')

        fig = generic_plot([w_hat_train, w_hat_val, w_hat_test, w_truth_train, w_truth_val, w_truth_test], w_pl_title,
                           w_pl_path, y_label="Deaths", formatter=format_xtick)
        summary.add_figure("final/deaths_fit", figure=fig)

        summary.flush()


    print(logged_info)
    print(best_epoch)


def get_exp_prefix(area, beta_t0, gamma_t0, delta_t0, lr_b, lr_g, lr_d, lr_a, train_size, der_1st_reg,
                   der_2nd_reg, t_inc, use_alpha, y_loss_weight, val_len):
    return area[0] + "_b" + str(beta_t0) + "_g" + str(gamma_t0) + "_d" + str(delta_t0) + \
           "_lrb" + str(lr_b) + "_lrg" + str(lr_g) + "_lrd" + str(lr_d) + "_lra" + str(lr_a) + "_ts" + str(train_size) \
           + "_vl" + str(val_len) + "_st_der" + str(der_1st_reg) + "_nd_der" + str(der_2nd_reg) + "_t_inc" + str(t_inc) \
           + "_use_alpha" + str(use_alpha) \
           + "_y_loss_weight" + str(y_loss_weight)


if __name__ == "__main__":
    n_epochs = 2500
    region = "Lombardia"
    beta_t = 0.8
    gamma_t = 0.3
    delta_t = 0.02
    lr_b = 1e-4
    lr_g = 1e-5
    lr_d = 3e-6
    lr_a = 0.
    train_size = 45
    val_len = 20
    der_1st_reg = 1e6
    der_2nd_reg = 0.
    use_alpha = False
    y_loss_weight = 0
    t_inc = 1.

    m = 0.2
    a = 1.0
    b = 0.05

    integrator = Heun

    exp_prefix = get_exp_prefix(region, beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, lr_a, train_size, der_1st_reg,
                                der_2nd_reg, t_inc, use_alpha, y_loss_weight, val_len)
    print(region)
    exp(region, populations[region], beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, lr_a, n_epochs, name=region,
        train_size=train_size, val_len=val_len,
        der_1st_reg=der_1st_reg, der_2nd_reg=der_2nd_reg, use_alpha=use_alpha, y_loss_weight=y_loss_weight, t_inc=t_inc,
        #exp_prefix=exp_prefix,
        exp_prefix=f"new_test_{datetime.now().strftime('%B_%d_%Y_%H_%M_%S')}",
        integrator=integrator, m=m, a=a, b=b)
