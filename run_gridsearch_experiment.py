import os

import multiprocessing as mp
import itertools

from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment

from utils.multiprocess_utils import ProcessPool
from params.params import SidartheParamGenerator

N_PROCESSES = 10
N_PERTURBED_RUNS = 50

if __name__ == "__main__":
    region = "FR"
    n_epochs = 10000
    t_step = 1.0
    train_size = 180

    der_1st_regs = [1e5, 1e6, 1e7, 1e8]
    momentums = [True]
    ms = [0.125]
    ass = [0.0]
    bound_reg = [1e5]
    loss_type = "nrmse"
    d_ws, r_ws, t_ws, h_ws, e_ws = [2.0, 3.0, 4.0, 5.0], [1.0], [1.0], [1.0], [1.0]
    val_len = 30

    initial_params = {
        "alpha": [0.210] * 4 + [0.570] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * (train_size - 38),
        "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
        "gamma": [0.2] * 4 + [0.456] * 18 + [0.285] * 6 + [0.11] * 10 + [0.11] * (train_size - 38),
        "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
        "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * (train_size - 38),
        "theta": [0.371] * train_size,
        "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
        "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
        "mu": [0.017] * 22 + [0.008] * (train_size - 22),
        "nu": [0.027] * 22 + [0.015] * (train_size - 22),
        "tau": [0.02]*train_size,
        "lambda": [0.034] * 22 + [0.08] * (train_size - 22),
        # "lambda": [0.034],
        "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
        # "kappa": [0.017],
        "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
        "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
        # "rho": [0.034],
        "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * (train_size - 38),
        "phi": [0.02] * train_size,
        "chi": [0.02] * train_size
    }

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment

    process_pool = ProcessPool(N_PROCESSES)
    mp.set_start_method('spawn')

    initial_params_list = [initial_params]
    for r in range(N_PERTURBED_RUNS):
        random_gen = SidartheParamGenerator()
        random_gen.random_init(train_size, ranges="extended")
        initial_params_list.append(random_gen.params)

    for hyper_params in itertools.product(ms, ass, der_1st_regs, d_ws, r_ws, t_ws, h_ws, e_ws, momentums, initial_params_list):
        m, a, der_1st_reg, d_w, r_w, t_w, h_w, e_w, momentum, initial_params = hyper_params

        loss_weights = {
            "d_weight": d_w,
            "r_weight": r_w,
            "t_weight": t_w,
            "h_weight": h_w,
            "e_weight": e_w,
        }

        for k, v in loss_weights.items():
            loss_weights[k] = 0.002 * v

        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="fit_FR")
        process_pool.start(
            target=experiment.run_exp,
            kwargs={
                "initial_params": initial_params,
                "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                "model_params": {"der_1st_reg": der_1st_reg, "bound_reg": bound_reg, "bound_loss_type": "step"},
                "train_params": {"momentum": momentum, "m": m, "a": a},
                "loss_weights": loss_weights
            }
        )
        process_pool.wait_for_empty_slot(timeout=5)

        process_pool.wait_for_all()
