import os
import pylab as pl
# import matplotlib.pyplot as pl
import torch
import datetime
import numpy as np
import pandas as pd
from ast import literal_eval

from learning_models.torch_sir import SirEq
from utils.data_utils import select_data
from utils.visualization_utils import plot_sir_dynamic, generic_plot, Curve, format_xtick


def exp(region, population, beta, gamma, delta, name, train_size, use_alpha, t_inc):

    df_file = os.path.join(os.getcwd(), "dati-regioni", "dpc-covid19-ita-regioni.csv")
    # df_file = os.path.join(os.getcwd(), "train.csv")
    area = [region]  # list(df["denominazione_regione"].unique())
    area_col_name = "denominazione_regione"  # "Country/Region"
    value_col_name = "deceduti"  # "Fatalities"
    groupby_cols = ["data"]  # ["Date"]

    x_target, w_target = select_data(df_file, area, area_col_name, value_col_name, groupby_cols, file_sep=",")
    _, y_target = select_data(df_file, area, area_col_name, "totale_positivi", groupby_cols, file_sep=",")
    print(y_target[0])
    print(w_target)

    initial_len = len(y_target)
    tmp_y, tmp_w = [], []
    for i in range(len(y_target)):
        if y_target[i] > 0:
            tmp_y.append(y_target[i])
            tmp_w.append(w_target[i])
    y_target = tmp_y
    w_target = tmp_w

    # START_DATE = datetime.date(2020, 2, 24 + initial_len - len(y_target))

    # def format_xtick(n, v):
    #     return (START_DATE + datetime.timedelta(int(n))).strftime("%d %b")  # e.g. "24 Feb", "25 Feb", ...

    # creating folders, if necessary
    exp_path = os.path.join(os.getcwd(), "regioni", "plot_results")
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    exp_path = os.path.join(exp_path, name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    val_size = train_size + 10  # validation on the next seven days
    dataset_size = len(w_target)
    exp_prefix = area[0] + "_best"

    dy_params = {"t_start": 0, "t_end": train_size}

    # init parameters
    epsilon = y_target[0].item() / population
    epsilon_z = w_target[0].item() / population
    S0 = 1 - (epsilon + epsilon_z)
    I0 = epsilon
    S0 = S0 * population
    I0 = I0 * population
    Z0 = epsilon_z

    init_cond = (S0, I0, Z0)  # initialization of SIR parameters (Suscettible, Infected, Recovered)
    sir = SirEq(beta, gamma, delta, population, init_cond=init_cond)

    # sir, mse_losses, der_1st_losses, der_2nd_losses = SirEq.train(w_target=w_target, y_target=y_target, **dy_params)
    w_hat, y_hat, sol = sir.inference(torch.arange(dy_params["t_start"], 100, t_inc))
    train_slice = slice(dy_params["t_start"], int(train_size/t_inc), int(1/t_inc))
    val_slice = slice(int(train_size/t_inc), int(val_size/t_inc), int(1/t_inc))
    test_slice = slice(int(val_size/t_inc), int(dataset_size/t_inc), int(1/t_inc))
    dataset_slice = slice(dy_params["t_start"], int(dataset_size/t_inc), int(1/t_inc))
    w_hat_train, w_hat_val, w_hat_test = w_hat[train_slice], w_hat[val_slice], w_hat[test_slice]
    w_hat_dataset = w_hat[dataset_slice]
    y_hat_train, y_hat_val, y_hat_test = y_hat[train_slice], y_hat[val_slice], y_hat[test_slice]
    y_hat_dataset = y_hat[dataset_slice]

    # train_risk, train_w_risk, train_y_risk, _, _, _ = sir.loss(w_hat_train, w_target[dy_params["t_start"]:train_size],
    #                                                            y_hat_train, y_target[dy_params["t_start"]:train_size])
    #
    # validation_risk, validation_w_risk, validation_y_risk, _, _, _ = sir.loss(w_hat_val, w_target[dy_params["t_end"]:val_size],
    #                                                                           y_hat_val, y_target[dy_params["t_end"]:val_size])
    #
    # test_risk, test_w_risk, test_y_risk, _, _, _ = sir.loss(w_hat_test, w_target[val_size:dataset_size],
    #                                                         y_hat_test, y_target[val_size:dataset_size])
    #
    # dataset_risk, _, _, _, _, _ = sir.loss(w_hat_dataset, w_target[dy_params["t_start"]:dataset_size],
    #                                  y_hat_dataset, y_target[dy_params["t_start"]:dataset_size])

    test_mape = sir.mape(w_hat_test, w_target[val_size:dataset_size])
    print("MAPE in Test: ")
    print(test_mape.detach().numpy())

    # Plotting
    file_format = ".png"

    # BETA, GAMMA, DELTA
    pl_x = list(range(train_size))  # list(range(len(beta)))
    beta_pl = Curve(pl_x, sir.beta.detach().numpy(), '-g', "$\\beta$")
    # beta_pl = Curve(pl_x, [sir.beta.detach().numpy()]*train_size, '-g', "$\\beta$")
    # gamma_pl = Curve(pl_x, sir.gamma.detach().numpy(), '-r', "$\gamma$")
    gamma_pl = Curve(pl_x, [sir.gamma.detach().numpy()]*train_size, '-r', "$\gamma$")
    delta_pl = Curve(pl_x, [sir.delta.detach().numpy()]*train_size, '-b', "$\delta$")
    params_curves = [beta_pl, gamma_pl, delta_pl]

    if use_alpha:
        # alpha = np.concatenate([sir.alpha(sir.get_policy_code(t)).detach().numpy().reshape(1) for t in range(len(sir.beta))], axis=0)
        alpha = np.concatenate([sir.alpha(sir.get_policy_code(t)).detach().numpy().reshape(1) for t in range(train_size)], axis=0)
        alpha_pl = Curve(pl_x, alpha, '-', "$\\alpha$")
        beta_alpha_pl = Curve(pl_x, alpha * sir.beta.detach().numpy(), '-', "$\\alpha \cdot \\beta$")
        params_curves.append(alpha_pl)
        params_curves.append(beta_alpha_pl)

    bgd_pl_title = "$\\beta, \gamma, \delta$  ({}".format(str(area[0])) + str(")")

    print(sir.beta.shape)

    bgd_pl_path = os.path.join(exp_path, exp_prefix + "_bcd_over_time" + file_format)
    generic_plot(params_curves, bgd_pl_title, bgd_pl_path, formatter=format_xtick)

    # R0
    pl_x = list(range(len(beta)))
    if use_alpha:
        alpha = np.concatenate([sir.alpha(sir.get_policy_code(t)).detach().numpy().reshape(1) for t in range(len(sir.beta))], axis=0)
        r0_pl = Curve(pl_x, (alpha * sir.beta.detach().numpy())/sir.gamma.detach().numpy(), '-', label="$R_0$")
    else:
        r0_pl = Curve(pl_x, sir.beta.detach().numpy()/sir.gamma.detach().numpy(), '-', label="$R_0$")
    thresh_r0_pl = Curve(pl_x, [1.0]*len(pl_x), '-', color="magenta")
    r0_pl_title = '$R_0$  ({}'.format(str(area[0])) + str(")")
    r0_pl_path = os.path.join(exp_path, exp_prefix + "_r0" + file_format)

    generic_plot([r0_pl, thresh_r0_pl], r0_pl_title, r0_pl_path, formatter=format_xtick)

    # normalize wrt population
    w_hat = w_hat[dataset_slice].detach().numpy() / population
    RES = sol.detach().numpy() / population
    print(len(RES[:,0]))
    _y = [_v / population for _v in y_target]
    _w = [_v / population for _v in w_target]

    # SIR dynamic
    sir_dir_path = os.path.join(exp_path, exp_prefix + "_SIR_global" + file_format)
    plot_sir_dynamic(RES[:, 0], RES[:, 1], RES[:, 2], area[0], sir_dir_path)

    # Infectious Train/Test/Real
    y_fits = Curve(list(range(train_size)), RES[:train_size, 1], '-r', label='$y$ fit')
    y_val_preds = Curve(list(range(train_size, val_size)), RES[train_size:val_size, 1], '-r', color='darkblue', label='$y$ validation')
    y_test_preds = Curve(list(range(val_size, dataset_size)), RES[val_size:dataset_size, 1], '-r', color='orange', label='$y$ prediction')
    y_truth_train = Curve(list(range(train_size)), _y[:train_size], '.r', label='$\hat{y}$ fit')
    y_truth_val = Curve(list(range(train_size, val_size)), _y[train_size:val_size], '.', color="darkblue", label='$\hat{y}$ validation')
    y_truth_test = Curve(list(range(val_size, dataset_size)), _y[val_size:dataset_size], '.', color="orange", label='$\hat{y}$ prediction')
    infected_pl_title = 'Infectious  ({}'.format(str(area[0])) + str(")")
    infected_pl_path = os.path.join(exp_path, exp_prefix + "_I_fit" + file_format)
    generic_plot([y_fits, y_val_preds, y_test_preds, y_truth_train, y_truth_val, y_truth_test], infected_pl_title, infected_pl_path, y_label='Infectious', formatter=format_xtick)

    # Deaths Train/Test/Real
    w_fits = Curve(list(range(train_size)), w_hat[:train_size], '-', label='$w$ fit')
    w_val_preds = Curve(list(range(train_size, val_size)), w_hat[train_size:val_size], '-', color='darkblue', label='$w$ validation')
    w_test_preds = Curve(list(range(val_size, dataset_size)), w_hat[val_size:dataset_size], '-', color='orange', label='$w$ prediction')
    w_truth_train = Curve(list(range(train_size)), _w[:train_size], '.r', label='$\hat{w}$ fit')
    w_truth_val = Curve(list(range(train_size, val_size)), _w[train_size:val_size], '.', color="darkblue", label='$\hat{w}$ validation')
    w_truth_test = Curve(list(range(val_size, dataset_size)), _w[val_size:dataset_size], '.', color="orange", label='$\hat{w}$ prediction')
    w_pl_title = 'Deaths  ({}'.format(str(area[0])) + str(")")
    w_pl_path = os.path.join(exp_path, exp_prefix + "_W_fit" + file_format)
    generic_plot([w_fits, w_val_preds, w_test_preds, w_truth_train, w_truth_val, w_truth_test], w_pl_title, w_pl_path, y_label="Deaths", formatter=format_xtick)


if __name__ == "__main__":
    population = {"Lombardia": 1e7, "Emilia-Romagna": 4.45e6, "Veneto": 4.9e6, "Piemonte": 4.36e6,
                  "Toscana": 3.73e6, "Umbria": 0.882e6, "Lazio": 5.88e6, "Marche": 1.525e6, "Campania": 5.802e6,
                  "Puglia": 1.551e6,
                  "Liguria": 4.029e6}
    exp_name = 'Marche_b0.9_g0.25_d0.02_lrb0.0001_lrg1e-05_lrd3e-06_ts40_st_der-1.0_nd_der-1.0_t_inc1.0_use_alphaFalse_y_loss_weight0.0'
    vals = exp_name.split("_")
    region = vals[0]
    exp_folder = "torch_sir_validation_only_beta_t"
    score_csv_path = os.path.join(os.getcwd(), "regioni", exp_folder, region, "scores.csv")
    df = pd.read_csv(score_csv_path, sep="\t")
    best_exp = df[df.name == exp_name].index.values[0]
    print(best_exp)
    print(literal_eval(df["beta"].iloc[best_exp]))
    print(literal_eval(df["gamma"].iloc[best_exp]))
    print(literal_eval(df["delta"].iloc[best_exp]))
    exp(region, population[region], literal_eval(df["beta"].iloc[best_exp]), literal_eval(df["gamma"].iloc[best_exp]),
        literal_eval(df["delta"].iloc[best_exp]), name=region,
        train_size=df["train_size"].iloc[best_exp], use_alpha=df["use_alpha"].iloc[best_exp], t_inc=df["t_inc"].iloc[best_exp])
