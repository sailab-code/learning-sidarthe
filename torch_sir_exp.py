from learning_models.torch_sir import SirEq
from utils.data_utils import select_data
import os
import pylab as pl
import torch
import numpy as np

def exp(region, population, beta_t0, gamma_t0, delta_t0, lr_b, lr_g, lr_d, n_epochs, name, train_size):

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

    # creating folders, if necessary
    base_path = os.path.join(os.getcwd(), "regioni")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    exp_path = os.path.join(base_path, "torch_sir")
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    exp_path = os.path.join(exp_path, name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    dataset_size = len(w_target)
    exp_prefix = area[0] + "_b" + str(beta_t0) + "_g" + str(gamma_t0) + "_d" + str(delta_t0) + \
                 "_lrb" + str(lr_b) + "_lrg" + str(lr_g) + "_lrd" + str(lr_d)

    minus = 0
    beta = [beta_t0 for _ in range(train_size - minus)]
    gamma = [gamma_t0 for _ in range(train_size - minus)]
    delta = [delta_t0 for _ in range(train_size - minus)]


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

    dy_params = {
        "beta": beta, "gamma": gamma, "delta": delta, "n_epochs": n_epochs,
         "population": population,
         "t_start": 0, "t_end": train_size,
         "lr_b": lr_b, "lr_g": lr_g, "lr_d": lr_d
    }

    sir = SirEq.train(target=w_target, y0 = y_target[0], z0=0., **dy_params)
    w_hat, sol = sir.inference(torch.arange(dy_params["t_start"], max(100, dataset_size)))
    train_risk, _ = sir.loss(w_hat[dy_params["t_start"]+1:train_size+1], w_target[dy_params["t_start"]:train_size])
    dataset_risk, _ = sir.loss(w_hat[dy_params["t_start"]+1:dataset_size+1], w_target[dy_params["t_start"]:dataset_size])

    log_file = os.path.join(exp_path, exp_prefix + "sir_" + area[0] + "_results.txt")
    with open(log_file, "w") as f:
        f.write("Beta:\n ")
        f.write(str(list(sir.beta)) + "\n")
        f.write("Gamma:\n ")
        f.write(str(list(sir.gamma)) + "\n")
        f.write("Delta:\n ")
        f.write(str(list(sir.delta)) + "\n")
        f.write("Train Risk:\n")
        f.write(str(train_risk) + "\n")
        f.write("Dataset Risk:\n")
        f.write(str(dataset_risk) + "\n")

    # BETA, GAMMA, DELTA plots
    fig, ax = pl.subplots()
    pl.title("Beta, Gamma, Delta over time")
    pl.grid(True)
    ax.plot(sir.beta.detach().numpy(), '-g', label="beta")
    ax.plot(sir.gamma.detach().numpy(), '-r', label="gamma")
    ax.plot(sir.delta.detach().numpy(), '-b', label="delta")
    ax.margins(0.05)
    ax.legend()
    pl.savefig(os.path.join(exp_path, exp_prefix + "bcd_over_time.png"))

    # normalize wrt population
    w_hat = w_hat.detach().numpy() / population
    RES = sol.detach().numpy() / population
    _y = [_v / population for _v in y_target]
    _w = [_v / population for _v in w_target]

    # Plotting
    pl.figure()
    pl.subplot(311)
    pl.grid(True)
    pl.title('SIR - Coronavirus in ' + str(area[0]))
    pl.plot(RES[:, 0], '-g', label='S')
    pl.legend(loc=0)
    pl.xlabel('Time in days')
    pl.ylabel('S')
    pl.subplot(312)
    pl.grid(True)
    pl.plot(RES[:, 1], '-r', label='I')
    pl.xlabel('Time in days')
    pl.ylabel('I')
    pl.subplot(313)
    pl.grid(True)
    pl.plot(RES[:, 2], '-k', label='R')
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
    pl.plot(list(range(train_size, dataset_size)), w_hat[train_size:dataset_size], '-', color='orange',
            label='Estimated Deaths')
    pl.plot(_w[:dataset_size], '.r', label='Deaths')
    pl.xlabel('Time in days')
    pl.ylabel('Deaths')
    pl.savefig(os.path.join(exp_path, exp_prefix + "sliding_W_fit.png"))
    pl.show()


if __name__ == "__main__":
    n_epochs = 2001
    region = "Lombardia"
    population = {"Lombardia": 1e7, "Emilia-Romagna": 4.45e6, "Veneto": 4.9e6, "Piemonte": 4.36e6}
    beta_t, gamma_t, delta_t = 0.81, 0.29, 0.03
    lr_b, lr_g, lr_d = 0.05, 0.01, 1e-7
    exp(region, population[region], beta_t, gamma_t, delta_t, lr_b, lr_g, lr_d, n_epochs, name="first_run", train_size=40)
