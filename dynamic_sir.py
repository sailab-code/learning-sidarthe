import os
import scipy.integrate as spi
import numpy as np
import pylab as pl
import skopt

from learning_models.differential_sliding_sir import SirEq
from utils.data_utils import select_data
from utils.visualization_utils import plot_data_and_fit


def exp(region, population, beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, n_epochs, fine_tune, learning_setup, is_static=True):
    """
    Single experiment run.
    :param beta_t: initial beta
    :param gamma_t: initial gamma
    :param delta_t: initial delta
    :param lr_b: learning rate of beta
    :param lr_g: learning rate of gamma
    :param lr_d: learning rate of delta
    :param n_epochs:
    :param learning_setup: possible ways for learning beta, gamma and delta:
    {all_window | last_only}
    :return:
    """

    # loading data
    df_file = os.path.join(os.getcwd(), "dati-regioni", "dpc-covid19-ita-regioni.csv")
    # df_file = os.path.join(os.getcwd(), "train.csv")
    area = [region]  # list(df["denominazione_regione"].unique())
    area_col_name = "denominazione_regione"  # "Country/Region"
    value_col_name = "deceduti"  # "Fatalities"
    groupby_cols = ["data"]  # ["Date"]

    _x, _w = select_data(df_file, area, area_col_name, value_col_name, groupby_cols, file_sep=",")
    _, _y = select_data(df_file, area, area_col_name, "totale_positivi", groupby_cols, file_sep=",")
    print(_y[0])
    print(_w)

    # creating folders, if necessary
    base_path = os.path.join(os.getcwd(), "regioni")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    exp_path = os.path.join(base_path, "sliding_sir")
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    name = learning_setup
    if is_static:
        name += "_static_joint_refine"

    if fine_tune:
        name += "_fine_tune"

    exp_path = os.path.join(exp_path, name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    train_size = 38
    dataset_size = len(_w)
    beta_t0, gamma_t0, delta_t0 = beta_t, gamma_t, delta_t

    exp_prefix = area[0] + "_b" + str(beta_t0) + "_g" + str(gamma_t0) + "_d" + str(delta_t0) +\
                 "_lrb" + str(lr_b) + "_lrg" + str(lr_g) + "_lrd" + str(lr_d)

    beta = [beta_t0 for _ in range(train_size)]
    gamma = [gamma_t0 for _ in range(train_size)]
    delta = [delta_t0 for _ in range(train_size)]

    # BETA, GAMMA, DELTA plots
    fig, ax = pl.subplots()
    pl.title("Beta, Gamma, Delta over time")
    pl.grid(True)
    ax.plot(beta, '-g', label="beta")
    ax.plot(gamma, '-r', label="gamma")
    ax.plot(delta, '-b', label="delta")
    ax.margins(0.05)
    ax.legend()
    pl.savefig(os.path.join(exp_path, exp_prefix + "initial_params_bcd_over_time.png"))

    # GLOBAL MODEL (No learning here just ode)
    dy_params = {"beta": beta, "gamma": gamma, "delta": delta, "n_epochs": n_epochs,
                 "population": population,
                 "t_start": 0, "t_end": train_size,
                 "lr_b": lr_b, "lr_g": lr_g, "lr_d": lr_d,
                 "eq_mode": "joint_dynamic",
                 "learning_setup": learning_setup}
    _, _, _, _, dynamic_sir, _ = SirEq.train(target=_w, y_0=_y[0], z_0=0.0, params=dy_params)  # configure dynamic_syr only
    RES, w_hat = dynamic_sir.inference(np.arange(dy_params["t_start"], max(100, dataset_size)), dynamic_sir.dynamic_bc_diff_eqs)  # run it on the first 100 days
    train_risk, _, _ = dynamic_sir.loss(np.arange(dy_params["t_start"], train_size), _w[dy_params["t_start"]:dy_params["t_end"]], dynamic_sir.dynamic_bc_diff_eqs)
    dataset_risk, _, _ = dynamic_sir.loss(np.arange(dy_params["t_start"], dataset_size), _w[dy_params["t_start"]:dataset_size], dynamic_sir.dynamic_bc_diff_eqs)

    log_file = os.path.join(exp_path, exp_prefix + "sir_" + area[0] + "_results.txt")
    with open(log_file, "w") as f:
        f.write("Beta:\n ")
        f.write(str(list(dynamic_sir.beta)) + "\n")
        f.write("Gamma:\n ")
        f.write(str(list(dynamic_sir.gamma)) + "\n")
        f.write("Delta:\n ")
        f.write(str(list(dynamic_sir.delta)) + "\n")
        f.write("Train Risk:\n")
        f.write(str(train_risk) + "\n")
        f.write("Dataset Risk:\n")
        f.write(str(dataset_risk) + "\n")

    # BETA, GAMMA, DELTA plots
    fig, ax = pl.subplots()
    pl.title("Beta, Gamma, Delta over time")
    pl.grid(True)
    ax.plot(dynamic_sir.beta, '-g', label="beta")
    ax.plot(dynamic_sir.gamma, '-r', label="gamma")
    ax.plot(dynamic_sir.delta, '-b', label="delta")
    ax.margins(0.05)
    ax.legend()
    pl.savefig(os.path.join(exp_path, exp_prefix + "bcd_over_time.png"))

    # normalize wrt population
    w_hat = w_hat/population
    RES = RES/population
    _y = [_v/population for _v in _y]
    _w = [_v/population for _v in _w]

    # Plotting
    pl.figure()
    pl.subplot(311)
    pl.grid(True)
    pl.title('SIR - Coronavirus in ' + str(area[0]))
    pl.plot(RES[:,0], '-g', label='S')
    pl.legend(loc=0)
    pl.xlabel('Time in days')
    pl.ylabel('S')
    pl.subplot(312)
    pl.grid(True)
    pl.plot(RES[:,1], '-r', label='I')
    pl.xlabel('Time in days')
    pl.ylabel('I')
    pl.subplot(313)
    pl.grid(True)
    pl.plot(RES[:,2], '-k', label='R')
    pl.xlabel('Time in days')
    pl.ylabel('R')
    pl.savefig(os.path.join(exp_path, exp_prefix + "sliding_SIR_global.png"))

    pl.figure()
    pl.grid(True)
    pl.title("Estimated Deaths")
    pl.plot(w_hat, '-', label='Estimated Deaths')
    pl.xlabel('Time in days')
    pl.ylabel('Deaths')
    pl.savefig(os.path.join(exp_path, exp_prefix + "sliding_W_global.png"))

    pl.figure()
    pl.grid(True)
    pl.title('SIR on fit')
    pl.plot(RES[:train_size, 1], '-r', label='I')
    pl.plot(list(range(train_size, dataset_size)), RES[train_size:dataset_size, 1], '-r', color='orange', label='I')
    pl.plot(_y[:dataset_size], '.b', label='I')
    pl.xlabel('Time in days')
    pl.ylabel('Infectious')
    pl.savefig(os.path.join(exp_path, exp_prefix + "sliding_I_fit.png"))

    pl.figure()
    pl.grid(True)
    pl.title("Estimated Deaths on fit")
    pl.plot(w_hat[:train_size], '-', label='Estimated Deaths')
    pl.plot(list(range(train_size, dataset_size)), w_hat[train_size:dataset_size], '-', color='orange', label='Estimated Deaths')
    pl.plot(_w[:dataset_size], '.r', label='Deaths')
    pl.xlabel('Time in days')
    pl.ylabel('Deaths')
    pl.savefig(os.path.join(exp_path, exp_prefix + "sliding_W_fit.png"))
    pl.show()


if __name__ == "__main__":
    learning_setup = "all_window"  # last_only
    n_epochs = 2501
    region = "Emilia-Romagna"
    population = 4.46e6
    beta_t, gamma_t, delta_t = 0.81, 0.29, 0.03
    lr_b, lr_g, lr_d = 2e-1, 4e-2, 2e-4
    fine_tune = True
    exp(region, population, beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, n_epochs, fine_tune, learning_setup, is_static=True)
