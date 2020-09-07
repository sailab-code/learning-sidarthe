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


import multiprocessing as mp
from utils.report_utils import get_exp_prefix, get_description, get_exp_description_html, get_markdown_description

from sidarthe_exp import exp

verbose = False
normalize = False

if __name__ == "__main__":
    n_epochs = 4000
    region = "UK"
    params = {
        "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * 44,
        "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 43),
        "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 10 + [0.11] * 44,
        "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 43),
        "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * 44,
        "theta": [0.371],
        "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * 44,
        "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * 44,
        "mu": [0.017] * 22 + [0.008] * (17+43),
        "nu": [0.027] * 22 + [0.015] * (17+43),
        "tau": [0.01],
        "lambda": [0.034] * 22 + [0.08] * (17+43),
        "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * 44,
        "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * 44,
        "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * 44,
        "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * 44
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
        "tau": 1e-7,
        "lambda": 1e-5,
        "kappa": 1e-5,
        "zeta": 1e-5,
        "rho": 1e-5,
        "sigma": 1e-5
    }

    # region extract reference

    # extract from csv with nature reference data

    references = {}
    ref_df = pd.read_csv(os.path.join(os.getcwd(), "regioni/sidarthe_results_new.csv"))
    for key in 'sidarthe':
        references[key] = ref_df[key].tolist()

    for key in ["r0", "h_detected"]:
        references[key] = ref_df[key].tolist()

    for key in params.keys():
        references[key] = ref_df[key].tolist()

    # endregion

    runs_directory = "runs_110_fit_e_2"
    train_size = 110
    val_len = 40
    der_1st_regs = [4.1e4]
    der_2nd_reg = 0.
    t_inc = 1.

    momentums = [False, True]
    ms = [1/8, 1/5, 1/2]
    ass = [0.04, 0.05, 0.1]

    bound_reg = 1e4

    integrator = Heun

    loss_type = "rmse"
    d_ws, r_ws, t_ws, h_ws = [0.1], [0.1], [1.0], [0.1]
    e_w = 5.0
    no_momentum_done = False
    procs = []
    mp.set_start_method('spawn')
    for hyper_params in itertools.product(ms, ass, der_1st_regs, d_ws, r_ws, t_ws, h_ws, momentums):
        m, a, der_1st_reg, d_w, r_w, t_w, h_w, momentum = hyper_params

        #do momentum=False only once (TODO: remove this when doing grid on other params)
        if momentum is False and no_momentum_done is True:
            continue

        if momentum is True:
            no_momentum_done = True

        loss_weights = {
            "d_weight": d_w,
            "r_weight": r_w,
            "t_weight": t_w,
            "h_weight": h_w,
            "e_weight": e_w,
        }

        """
        exp(region, populations[region], params,
                            learning_rates, n_epochs, region, train_size, val_len,
                            loss_weights, der_1st_reg, bound_reg, t_inc, integrator,
                            momentum, m, a, loss_type, references, runs_directory
            )
        """

        proc = mp.Process(target=exp,
                          args=(region, populations[region], params,
                            learning_rates, n_epochs, region, train_size, val_len,
                            loss_weights, der_1st_reg, bound_reg, t_inc, integrator,
                            momentum, m, a, loss_type, references, runs_directory)
                          )

        proc.start()
        procs.append(proc)

        # run 6 exps at a time
        if len(procs) == 10:
            for proc in procs:
                proc.join()
            procs.clear()

    for proc in procs:
        proc.join()