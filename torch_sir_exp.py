import os
import pylab as pl
# import matplotlib.pyplot as pl
import torch
import datetime

from learning_models.torch_sir import SirEq
from utils.data_utils import select_data
from utils.visualization_utils import plot_sir_dynamic, generic_plot, Curve, format_xtick


def exp(region, population, beta_t0, gamma_t0, delta_t0, lr_b, lr_g, lr_d, n_epochs, name, train_size, derivative_reg, der_2nd_reg):

    df_file = os.path.join(os.getcwd(), "COVID-19", "dati-regioni", "dpc-covid19-ita-regioni.csv")
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

    exp_path = os.path.join(base_path, "torch_sir_ellis")
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    exp_path = os.path.join(exp_path, name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    dataset_size = len(w_target)
    exp_prefix = area[0] + "_b" + str(beta_t0) + "_g" + str(gamma_t0) + "_d" + str(delta_t0) + \
                 "_lrb" + str(lr_b) + "_lrg" + str(lr_g) + "_lrd" + str(lr_d) + "_ts" + str(train_size) \
                 + "_st_der" + str(derivative_reg) + "_nd_der" + str(der_2nd_reg)

    minus = 0
    beta = [beta_t0 for _ in range(train_size - minus)]
    gamma = [gamma_t0 for _ in range(train_size - minus)]
    delta = [delta_t0 for _ in range(train_size - minus)]

    # BETA, GAMMA, DELTA plots
    pl_x = list(range(len(beta)))
    beta_pl = Curve(pl_x, beta, '-g', "$\\beta$")
    gamma_pl = Curve(pl_x, gamma, '-r', "$\gamma$")
    delta_pl = Curve(pl_x, [delta] * train_size, '-b', "$\delta$")

    bgd_pl_title = "$\\beta, \gamma, \delta$  ({}".format(str(area[0])) + str(")")

    bgd_pl_path = os.path.join(exp_path, exp_prefix + "initial_params_bcd_over_time.pdf")
    generic_plot([beta_pl, gamma_pl, delta_pl], bgd_pl_title, bgd_pl_path, formatter=format_xtick)

    dy_params = {
        "beta": beta, "gamma": gamma, "delta": delta, "n_epochs": n_epochs,
         "population": population,
         "t_start": 0, "t_end": train_size,
         "lr_b": lr_b, "lr_g": lr_g, "lr_d": lr_d,
        "derivative_reg": derivative_reg,
        "der_2nd_reg": der_2nd_reg
    }

    sir, losses = SirEq.train(target=w_target, y0 = y_target[0], z0=0., **dy_params)
    w_hat, sol = sir.inference(torch.arange(dy_params["t_start"], max(100, dataset_size)))
    train_risk, _ = sir.loss(w_hat[dy_params["t_start"]:train_size], torch.tensor(w_target[dy_params["t_start"]:train_size], dtype=torch.float32))
    dataset_risk, _ = sir.loss(w_hat[dy_params["t_start"]:dataset_size], torch.tensor(w_target[dy_params["t_start"]:dataset_size], dtype=torch.float32))

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
        f.write(str(losses) + "\n")

    csv_file = os.path.join(exp_path, "scores.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("name\tbeta_t0\tgamma_t0\tdelta_t0\tbeta\tgamma\tdelta\tlr_beta\tlr_gamma\tlr_delta\ttrain_size\tfirst_derivative_reg\tsecond_derivative_reg\ttrain_risk\tdataset_risk\n")

    with open(csv_file, "a") as f:
        _res_str = '\t'.join(
            [exp_prefix, str(beta_t0), str(gamma_t0), str(delta_t0),
             str(list(sir.beta.detach().numpy())).replace("\n", " "), str(list(sir.gamma.detach().numpy())).replace("\n", " "),
             str(list(sir.delta.detach().numpy())).replace("\n", " "),
             str(dy_params["lr_b"]), str(dy_params["lr_g"]), str(dy_params["lr_d"]), str(train_size),
             str(derivative_reg), str(der_2nd_reg),
             str(train_risk.detach().numpy()), str(dataset_risk.detach().numpy()) + "\n"])
        f.write(_res_str)

    # Plotting

    # BETA, GAMMA, DELTA
    pl_x = list(range(len(beta)))
    beta_pl = Curve(pl_x, sir.beta.detach().numpy(), '-g', "$\\beta$")
    gamma_pl = Curve(pl_x, sir.gamma.detach().numpy(), '-r', "$\gamma$")
    delta_pl = Curve(pl_x, [sir.delta.detach().numpy()]*train_size, '-b', "$\delta$")

    bgd_pl_title = "$\\beta, \gamma, \delta$  ({}".format(str(area[0])) + str(")")

    bgd_pl_path = os.path.join(exp_path, exp_prefix + "bcd_over_time.pdf")
    generic_plot([beta_pl, gamma_pl, delta_pl], bgd_pl_title, bgd_pl_path, formatter=format_xtick)

    # R0
    pl_x = list(range(len(beta)))
    r0_pl = Curve(pl_x, sir.beta.detach().numpy()/sir.gamma.detach().numpy(), '-', label="$R_0$")
    thresh_r0_pl = Curve(pl_x, [1.0]*len(pl_x), '-', color="magenta")

    r0_pl_title = '$R_0$  ({}'.format(str(area[0])) + str(")")
    r0_pl_path = os.path.join(exp_path, exp_prefix + "r0.pdf")

    generic_plot([r0_pl, thresh_r0_pl], r0_pl_title, r0_pl_path, formatter=format_xtick)

    # Risk
    risk_pl = Curve(range(0, len(losses)*50, 50), losses, '-b', label="risk")
    risk_pl_title = "Risk over epochs  ({}".format(str(area[0])) + str(")")
    risk_pl_path = os.path.join(exp_path, exp_prefix + "risk_over_epochs.pdf")
    generic_plot([risk_pl], risk_pl_title, risk_pl_path)

    # normalize wrt population
    w_hat = w_hat.detach().numpy() / population
    RES = sol.detach().numpy() / population
    _y = [_v / population for _v in y_target]
    _w = [_v / population for _v in w_target]

    # SIR dynamic
    sir_dir_path = os.path.join(exp_path, exp_prefix + "sliding_SIR_global.pdf")
    plot_sir_dynamic(RES[:, 0], RES[:, 1], RES[:, 2], area[0], sir_dir_path)

    # Deaths
    pl_w_hat = list(range(len(w_hat)))
    w_hat_pl = Curve(pl_w_hat, w_hat, '-', label='$w$')

    deaths_pl_title = 'Deaths  ({}'.format(str(area[0])) + str(")")
    deaths_pl_path = os.path.join(exp_path, exp_prefix + "sliding_W_global.pdf")

    generic_plot([w_hat_pl], deaths_pl_title, deaths_pl_path, 'Time in days', 'Deaths')

    # Infectious Train/Test/Real
    y_fits = Curve(list(range(train_size)), RES[:train_size, 1], '-r', label='$y$ fit')
    y_preds = Curve(list(range(train_size, dataset_size)), RES[train_size:dataset_size, 1], '-r', color='orange', label='$y$ prediction')
    y_hats = Curve(list(range(dataset_size)), _y[:dataset_size], '.b', label='$\hat{y}$')

    infected_pl_title = 'Infectious  ({}'.format(str(area[0])) + str(")")
    infected_pl_path = os.path.join(exp_path, exp_prefix + "sliding_I_fit.pdf")

    generic_plot([y_fits, y_preds, y_hats], infected_pl_title, infected_pl_path, y_label='Infectious', formatter=format_xtick)

    # Deaths Train/Test/Real
    w_fits = Curve(list(range(train_size)), w_hat[:train_size], '-', label='$w$ fit')
    w_preds = Curve(list(range(train_size, dataset_size)), w_hat[train_size:dataset_size], '-', color='orange', label='$w$ prediction')
    w_hats = Curve(list(range(dataset_size)), _w[:dataset_size], '.r', label='$\hat{w}$')

    w_pl_title = 'Deaths  ({}'.format(str(area[0])) + str(")")
    w_pl_path = os.path.join(exp_path, exp_prefix + "sliding_W_fit.pdf")

    generic_plot([w_fits, w_preds, w_hats], w_pl_title, w_pl_path, y_label="Deaths", formatter=format_xtick)


if __name__ == "__main__":
    n_epochs = 501
    # Veneto b0.8_g0.35_d0.015_lrb0.05_lrg0.01_lrd0.0005_ts40_st_der1000.0_nd_der0.0
    # Lombardia b0.81_g0.2_d0.02_lrb0.05_lrg0.01_lrd1e-05_ts40_st_der1000.0_nd_der10000.0
    # Emilia-Romagna b0.8_g0.35_d0.015_lrb0.05_lrg0.01_lrd5e-05_ts40_st_der1000.0_nd_der0.0
    regions = ["Lombardia"]
    population = {"Lombardia": 1e7, "Emilia-Romagna": 4.45e6, "Veneto": 4.9e6, "Piemonte": 4.36e6,
                  "Toscana": 3.73e6, "Umbria": 0.882e6, "Lazio": 5.88e6, "Marche": 1.525e6, "Campania": 5.802e6,
                  "Puglia": 1.551e6,
                  "Liguria": 4.029e6}
    beta_ts, gamma_ts, delta_ts = [0.37], [0.2], [0.025]
    lr_bs, lr_gs, lr_ds = [5e-2], [1e-2], [2e-5]
    train_sizes = list(range(40, 41, 5))
    derivative_regs = [1e3]  # [0.0, 1e2, 1e3]
    der_2nd_regs = [0.0]  # [0.0, 1e2, 1e3]

    import itertools
    for region, beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, train_size, derivative_reg,der_2nd_reg in itertools.product(regions, beta_ts, gamma_ts, delta_ts, lr_bs, lr_gs, lr_ds, train_sizes, derivative_regs, der_2nd_regs):
        print(region)
        exp(region, population[region], beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, n_epochs, name=region, train_size=train_size, derivative_reg=derivative_reg, der_2nd_reg=der_2nd_reg)
