import os
import pylab as pl
# import matplotlib.pyplot as pl
import torch
import datetime
import numpy as np

from learning_models.torch_sir import SirEq
from utils.data_utils import select_data
from utils.visualization_utils import plot_sir_dynamic, generic_plot, Curve, format_xtick


def exp(region, population, beta_t0, gamma_t0, delta_t0, lr_b, lr_g, lr_d, lr_a, n_epochs, name, train_size, der_1st_reg, der_2nd_reg, use_alpha, y_loss_weight, t_inc):

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
    base_path = os.path.join(os.getcwd(), "regioni")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    exp_path = os.path.join(base_path, "torch_sir_validation_only_beta_t_test_grafici")
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    exp_path = os.path.join(exp_path, name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    # t_inc = 1.0

    val_size = train_size + 10  # validation on the next seven days
    dataset_size = len(w_target)
    exp_prefix = area[0] + "_b" + str(beta_t0) + "_g" + str(gamma_t0) + "_d" + str(delta_t0) + \
                 "_lrb" + str(lr_b) + "_lrg" + str(lr_g) + "_lrd" + str(lr_d) + "_ts" + str(train_size) \
                 + "_st_der" + str(der_1st_reg) + "_nd_der" + str(der_2nd_reg) + "_t_inc" + str(t_inc) + "_use_alpha" + str(use_alpha) \
                 + "_y_loss_weight" + str(y_loss_weight)

    beta = [beta_t0 for _ in range(int(train_size))]
    # beta = [beta_t0]
    # gamma = [gamma_t0 for _ in range(int(train_size))]
    gamma = [gamma_t0]
    delta = [delta_t0]

    dy_params = {
        "beta": beta, "gamma": gamma, "delta": delta, "n_epochs": n_epochs,
        "population": population,
        "t_start": 0, "t_end": train_size,
        "lr_b": lr_b, "lr_g": lr_g, "lr_d": lr_d, "lr_a": lr_a,
        "der_1st_reg": der_1st_reg,
        "der_2nd_reg": der_2nd_reg,
        "t_inc": t_inc,
        "momentum": True,
        "use_alpha": use_alpha,
        "y_loss_weight": y_loss_weight,
    }

    sir, mse_losses, der_1st_losses, der_2nd_losses = SirEq.train(w_target=w_target, y_target=y_target, **dy_params)
    w_hat, y_hat, sol = sir.inference(torch.arange(dy_params["t_start"], 100, t_inc))
    train_slice = slice(dy_params["t_start"], int(train_size/t_inc), int(1/t_inc))
    val_slice = slice(int(train_size/t_inc), int(val_size/t_inc), int(1/t_inc))
    test_slice = slice(int(val_size/t_inc), int(dataset_size/t_inc), int(1/t_inc))
    dataset_slice = slice(dy_params["t_start"], int(dataset_size/t_inc), int(1/t_inc))
    w_hat_train, w_hat_val, w_hat_test = w_hat[train_slice], w_hat[val_slice], w_hat[test_slice]
    w_hat_dataset = w_hat[dataset_slice]
    y_hat_train, y_hat_val, y_hat_test = y_hat[train_slice], y_hat[val_slice], y_hat[test_slice]
    y_hat_dataset = y_hat[dataset_slice]

    train_risk, train_w_risk, train_y_risk, _, _, _ = sir.loss(w_hat_train, w_target[dy_params["t_start"]:train_size],
                                                               y_hat_train, y_target[dy_params["t_start"]:train_size])

    validation_risk, validation_w_risk, validation_y_risk, _, _, _ = sir.loss(w_hat_val, w_target[dy_params["t_end"]:val_size],
                                                                              y_hat_val, y_target[dy_params["t_end"]:val_size])

    test_risk, test_w_risk, test_y_risk, _, _, _ = sir.loss(w_hat_test, w_target[val_size:dataset_size],
                                                            y_hat_test, y_target[val_size:dataset_size])

    dataset_risk, _, _, _, _, _ = sir.loss(w_hat_dataset, w_target[dy_params["t_start"]:dataset_size],
                                     y_hat_dataset, y_target[dy_params["t_start"]:dataset_size])

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
        f.write("Dataset Risk:\n")
        f.write(str(dataset_risk.detach().numpy()) + "\n")
        f.write("Loss over epochs: \n")
        f.write(str(mse_losses) + "\n")

    csv_file = os.path.join(exp_path, "scores.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("name\tbeta_t0\tgamma_t0\tdelta_t0\tbeta\tgamma\tdelta\tlr_beta\tlr_gamma\tlr_delta\t"
                    "train_size\tfirst_derivative_reg\tsecond_derivative_reg\tuse_alpha\ty_loss_weight\tt_inc\t"
                    "w_train_risk\tw_val_risk\tw_test_risk\t"
                    "train_risk\tval_risk\ttest_risk\tdataset_risk\n")

    with open(csv_file, "a") as f:
        _res_str = '\t'.join(
            [exp_prefix, str(beta_t0), str(gamma_t0), str(delta_t0),
             str(list(sir.beta.detach().numpy())).replace("\n", " "), str(list(sir.gamma.detach().numpy())).replace("\n", " "),
             str(list(sir.delta.detach().numpy())).replace("\n", " "),
             str(dy_params["lr_b"]), str(dy_params["lr_g"]), str(dy_params["lr_d"]), str(train_size),
             str(der_1st_reg), str(der_2nd_reg), str(use_alpha), str(y_loss_weight), str(t_inc),
             str(train_w_risk.detach().numpy()), str(validation_w_risk.detach().numpy()), str(test_w_risk.detach().numpy()),
             str(train_risk.detach().numpy()), str(validation_risk.detach().numpy()), str(test_risk.detach().numpy()), str(dataset_risk.detach().numpy()) + "\n"])
        f.write(_res_str)

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

    # Risk
    # risk_pl = Curve(range(0, len(mse_losses)*50, 50), mse_losses, '-b', label="risk")
    # risk_pl_title = "Risk over epochs  ({}".format(str(area[0])) + str(")")
    # risk_pl_path = os.path.join(exp_path, exp_prefix + "_risk" + file_format)
    # generic_plot([risk_pl], risk_pl_title, risk_pl_path)

    # normalize wrt population
    w_hat = w_hat[dataset_slice].detach().numpy() / population
    RES = sol.detach().numpy() / population
    print(len(RES[:,0]))
    _y = [_v / population for _v in y_target]
    _w = [_v / population for _v in w_target]

    # SIR dynamic
    sir_dir_path = os.path.join(exp_path, exp_prefix + "_SIR_global" + file_format)
    plot_sir_dynamic(RES[:, 0], RES[:, 1], RES[:, 2], area[0], sir_dir_path)

    # Deaths
    pl_w_hat = list(range(len(w_hat)))
    w_hat_pl = Curve(pl_w_hat, w_hat, '-', label='$w$')

    deaths_pl_title = 'Deaths  ({}'.format(str(area[0])) + str(")")
    deaths_pl_path = os.path.join(exp_path, exp_prefix + "_W_global" + file_format)

    generic_plot([w_hat_pl], deaths_pl_title, deaths_pl_path, 'Time in days', 'Deaths')

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
    # todo provare ad inizializzare rete con pesi random
    n_epochs = 101
    # Veneto b0.8_g0.35_d0.015_lrb0.05_lrg0.01_lrd0.0005_ts40_st_der1000.0_nd_der0.0
    # Lombardia b0.81_g0.2_d0.02_lrb0.05_lrg0.01_lrd1e-05_ts40_st_der1000.0_nd_der10000.0
    # Emilia-Romagna b0.8_g0.35_d0.015_lrb0.05_lrg0.01_lrd5e-05_ts40_st_der1000.0_nd_der0.0
    regions = ["Marche", "Umbria", "Toscana", "Lazio"]  # ["Lombardia", "Emilia-Romagna", "Veneto", "Piemonte",  "Toscana", "Umbria", "Lazio", "Marche", "Campania","Puglia", "Liguria"]

    population = {"Lombardia": 1e7, "Emilia-Romagna": 4.45e6, "Veneto": 4.9e6, "Piemonte": 4.36e6,
                  "Toscana": 3.73e6, "Umbria": 0.882e6, "Lazio": 5.88e6, "Marche": 1.525e6, "Campania": 5.802e6,
                  "Puglia": 1.551e6,
                  "Liguria": 4.029e6}
    beta_ts, gamma_ts, delta_ts = [0.9, 0.75, 0.6], [0.35, 0.25], [0.005, 0.008, 0.0125, 0.02]
    lr_bs, lr_gs, lr_ds, lr_as = [1e-4], [1e-5], [3e-6], [1e-3]
    train_sizes = [40]  # list(range(40, 41, 5))
    derivative_regs = [-1.0]  # [0.0, 1e2, 1e3]
    der_2nd_regs = [-1.0]  # [0.0, 1e2, 1e3]
    use_alphas = [False]
    y_loss_weights = [0.0]
    t_incs = [1.0]

    import itertools
    for hyper_params in itertools.product(regions, beta_ts, gamma_ts, delta_ts, lr_bs, lr_gs, lr_ds, lr_as, train_sizes, derivative_regs, der_2nd_regs, use_alphas, y_loss_weights, t_incs):
        region, beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, lr_a, train_size, derivative_reg, der_2nd_reg, use_alpha, y_loss_w, t_inc = hyper_params
        print(region)
        exp(region, population[region], beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, lr_a, n_epochs, name=region, train_size=train_size,
            der_1st_reg=derivative_reg, der_2nd_reg=der_2nd_reg, use_alpha=use_alpha, y_loss_weight=y_loss_w, t_inc=t_inc)
