import os
import scipy.integrate as spi
import numpy as np
import pylab as pl
import skopt


from learning_models.differential_sliding_sir import SirEq
from utils.data_utils import select_data
from utils.visualization_utils import plot_data_and_fit


def exp(beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, ws, n_epochs, fine_tune, learning_setup, is_static=True):
    """
    Single experiment run.
    :param beta_t: initial beta
    :param gamma_t: initial gamma
    :param delta_t: initial delta
    :param lr_b: learning rate of beta
    :param lr_g: learning rate of gamma
    :param lr_d: learning rate of delta
    :param ws: sliding window's size
    :param n_epochs:
    :param learning_setup: possible ways for learning beta, gamma and delta:
    {all_window | last_only}
    :return:
    """

    # loading data
    df_file = os.path.join(os.getcwd(), "dati-regioni", "dpc-covid19-ita-regioni.csv")
    # df_file = os.path.join(os.getcwd(), "train.csv")
    area = ["Lombardia"]  # list(df["denominazione_regione"].unique())
    area_col_name = "denominazione_regione"  # "Country/Region"
    value_col_name = "deceduti"  # "Fatalities"
    groupby_cols = ["data"]  # ["Date"]

    _x, _w = select_data(df_file, area, area_col_name, value_col_name, groupby_cols, file_sep=",")
    _, _y = select_data(df_file, area, area_col_name, "totale_positivi", groupby_cols, file_sep=",")
    print(_y[0])

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

    population = 1e7
    train_size = 38
    beta, gamma, delta = [], [], []
    risks = []
    params = None
    res = None
    beta_t0, gamma_t0, delta_t0 = beta_t, gamma_t, delta_t

    # initialize beta,gamma,delta for the first window (the same value)
    params_0 = {"beta": [beta_t0], "gamma": [gamma_t0], "delta": [delta_t0], "n_epochs": n_epochs,
                "population": population,
                "t_start": 0, "t_end": ws+1,
                "lr_b": lr_b, "lr_g": lr_g, "lr_d": lr_d,
                "eq_mode": "static",
                "learning_setup": "all_window"}
    # loss, beta_t, gamma_t, delta_t, _, _ = SirEq.train(target=_w, y_0=_y[0], z_0=0.0, params=params_0)
    # beta.extend([beta_t]*params_0["t_end"])
    # gamma.extend([gamma_t]*params_0["t_end"])
    # delta.extend([delta_t]*params_0["t_end"])
    beta.extend([beta_t0] * params_0["t_end"])
    gamma.extend([gamma_t0]*params_0["t_end"])
    delta.extend([delta_t0]*params_0["t_end"])

    # SLIDING WINDOW
    for i in range(ws+1, train_size):  # one_side window
        ST = i - ws-1
        ND = i+1  # number of days of simulation
        if not is_static:
            # start at time ST, learn a model with window_size*3 beta,gamma,delta, but only beta(ST), gamma(ST) and delta(ST) are learned.
            params = {"beta": beta[ST:i] + [beta[-1]], "gamma": gamma[ST:i] + [gamma[-1]], "delta": delta[ST:i]+[delta[-1]], "n_epochs": n_epochs,
                      "population": population,
                      "t_start": ST, "t_end": ND,
                      "lr_b": lr_b, "lr_g": lr_g, "lr_d": lr_d,
                      "eq_mode": "dynamic",
                      "learning_setup": learning_setup}
        else:
            params = {"beta": [beta[-1]], "gamma": [gamma[-1]], "delta": [delta[-1]], "n_epochs": n_epochs,
                      "population": population,
                      "t_start": ST, "t_end": ND,
                      "lr_b": lr_b, "lr_g": lr_g, "lr_d": lr_d,
                      "eq_mode": "static",
                      "learning_setup": learning_setup}

        print("Window: [%d, %d)" % (ST, ND))

        if params["t_start"] > 0:
            # set y_start and z_start with the previous window model predictions
            print(res)
            z_start = float(res[1, 2])
            y_start = float(res[1, 1])
            print(z_start)
            print(y_start)
        else:
            # initialization
            z_start = 0.0
            y_start = _y[0]

        loss, beta_t, gamma_t, delta_t, _, res = SirEq.train(target=_w, y_0=y_start, z_0=z_start, params=params)

        print("Final Loss: %.7f" % loss)

        beta.append(beta_t)
        gamma.append(gamma_t)
        delta.append(delta_t)
        risks.append(loss)

    # beta = [beta[0]]*ws + beta  # + [beta[-1]]*ws  # set early or late betas
    # gamma = [gamma[0]]*ws + gamma  # + [gamma[-1]]*ws  # set early or late gammas
    # delta = [delta[0]]*ws + delta  # + [delta[-1]]*ws  # set early or late deltas

    exp_prefix = area[0] + "_b" + str(beta_t0) + "_g" + str(gamma_t0) + "_d" + str(delta_t0) +\
                 "_lrb" + str(params["lr_b"]) + "_lrg" + str(params["lr_g"]) + "_lrd" + str(params["lr_d"]) + "_mono_sw%d_" % ws

    _x_range = np.arange(ws+1, train_size)
    pl.figure()
    pl.grid(True)
    pl.title("Risk over time")
    pl.plot(_x_range, risks, '-', label=' Risk')
    pl.xlabel('Time in days')
    pl.ylabel('MSE')
    pl.savefig(os.path.join(exp_path, exp_prefix + "risk.png"))

    log_file = os.path.join(exp_path, exp_prefix + "sir_" + area[0] + "_results.txt")
    with open(log_file, "w") as f:
        f.write("Beta:\n ")
        f.write(str(list(beta)) + "\n")
        f.write("Gamma:\n ")
        f.write(str(list(gamma)) + "\n")
        f.write("Delta:\n ")
        f.write(str(list(delta)) + "\n")
        f.write("Losses:\n ")
        f.write(str(list(risks)) + "\n")

    # MANUAL BETA fixme
    # beta = [0.49 - 0.0076*i for i in range(train_size)]
    # # beta[-10:] = [beta[-1] - 0.01*i for i in range(1,11)]
    # gamma = [0.185 + 0.0015*i for i in range(train_size)]
    # delta = [0.05 + 0.00057*i for i in range(train_size)]
    # import random
    #
    # beta = [random.uniform(0.2, 0.8) for _ in range(train_size)]
    # gamma = [random.uniform(0.05, 0.5) for _ in range(train_size)]
    # delta = [random.uniform(0.01, 0.06) for _ in range(train_size)]
    beta = [0.81 for _ in range(train_size)]
    gamma = [0.29 for _ in range(train_size)]
    delta = [0.03 for _ in range(train_size)]

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
    dy_params = {"beta": beta, "gamma": gamma, "delta": delta, "n_epochs": 4001 if fine_tune else 0,
                 "population": population,
                 "t_start": 0, "t_end": train_size,
                 "lr_b": 2e-1, "lr_g": 4e-2, "lr_d": 2e-4,
                 "eq_mode": "joint_dynamic",
                 # "eq_mode": "dynamic",
                 "learning_setup": learning_setup}
    _, _, _, _, dynamic_sir, _ = SirEq.train(target=_w, y_0=_y[0], z_0=0.0, params=dy_params)  # configure dynamic_syr only
    RES, w_hat = dynamic_sir.inference(np.arange(dy_params["t_start"], 100), dynamic_sir.dynamic_bc_diff_eqs)  # run it on the first 100 days
    global_risk, _, _ = dynamic_sir.loss(np.arange(dy_params["t_start"], train_size), _w[dy_params["t_start"]:dy_params["t_end"]], dynamic_sir.dynamic_bc_diff_eqs)

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

    # saving results into csv
    log_file = os.path.join(exp_path, "sir_scores.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("name\tlearning_setting\tbeta\tgamma\tdelta\tlr_beta\tlr_gamma\tlr_delta\tsingle_models_risk\tglobal_model_risk\n")

    with open(log_file, "a") as f:
        _res_str = '\t'.join([exp_prefix, learning_setup, str(beta), str(gamma), str(delta),
                              str(params["lr_b"]), str(params["lr_g"]), str(params["lr_d"]),
                              str(sum(list(risks))), str(global_risk) + "\n"])
        f.write(_res_str)

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
    pl.plot(_y[:train_size], '.b', label='I')
    pl.xlabel('Time in days')
    pl.ylabel('Infectious')
    pl.savefig(os.path.join(exp_path, exp_prefix + "sliding_I_fit.png"))

    pl.figure()
    pl.grid(True)
    pl.title("Estimated Deaths on fit")
    pl.plot(w_hat[:train_size], '-', label='Estimated Deaths')
    pl.plot(_w[:train_size], '.r', label='Deaths')
    pl.xlabel('Time in days')
    pl.ylabel('Deaths')
    pl.savefig(os.path.join(exp_path, exp_prefix + "sliding_W_fit.png"))
    pl.show()


if __name__ == "__main__":
    learning_setup = "all_window"  # last_only
    n_epochs = 10
    for ws in range(4, 5):
        beta_t, gamma_t, delta_t = 0.81, 0.29, 0.03
        lr_b, lr_g, lr_d = 2e-1, 4e-2, 2e-4
        fine_tune = True
        exp(beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, ws, n_epochs, fine_tune, learning_setup, is_static=True)
