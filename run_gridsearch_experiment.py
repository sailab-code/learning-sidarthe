import os

from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment

from learning_models.tied_sidarthe_extended import TiedSidartheExtended
from params.params import SidartheParamGenerator
from utils.multiprocess_utils import ProcessPool

import multiprocessing as mp
import itertools

N_PROCESSES = 6
N_PERTURBED_RUNS = 0

if __name__ == "__main__":
    region = "FR"
    n_epochs = 8000
    t_step = 1.0
    train_size = 182

    der_1st_regs = [5e5, 1e7]
    momentums = [True]

    ms = [0.1]
    ass = [0.0]
    bound_regs = [1.0, 2.5, 5.0]
    loss_type = "nrmse"
    d_ws, r_ws, t_ws, h_ws, e_ws = [0.0], [1.0, 2.0, 3.0], [1.0], [1.0], [1.0]
    val_len = 20

    # Italy
    # initial_params = {
    #         "alpha": [0.210] * 4 + [0.570] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * (train_size - 38),
    #         "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
    #         "gamma": [0.2] * 4 + [0.456] * 18 + [0.285] * 6 + [0.11] * 10 + [0.11] * (train_size - 38),
    #         "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (train_size - 22),
    #         "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * (train_size - 38),
    #         "theta": [0.371] * train_size,
    #         "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
    #         "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * (train_size - 38),
    #         "mu": [0.017] * 22 + [0.008] * (train_size - 22),
    #         "nu": [0.027] * 22 + [0.015] * (train_size - 22),
    #         "tau": [0.05]*train_size,
    #         "lambda": [0.034] * 22 + [0.08] * (train_size - 22),
    #         "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
    #         "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
    #         "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * (train_size - 38),
    #         "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * (train_size - 38),
    #         "phi": [0.02] * train_size,
    #         "chi": [0.02] * train_size
    #     }

    # France
    initial_params = {
        "alpha": [0.25] * train_size,
        "beta": [0.0055] * train_size,
        "gamma": [0.2] * train_size,
        "delta": [0.0057] * train_size,
        "epsilon": [0.143] * train_size,
        "theta": [0.371] * train_size,
        "zeta": [0.0025] * train_size,
        "eta": [0.0025] * train_size,
        "mu": [0.008] * train_size ,
        "nu": [0.015] * train_size,
        "tau": [0.02]*train_size,
        "lambda": [0.07] * train_size,
        "kappa": [0.018] * train_size,
        "xi": [0.018] * train_size,
        "rho": [0.018] * train_size,
        "sigma": [0.01] * train_size,
        "phi": [0.02] * train_size,
        "chi": [0.02] * train_size
    }

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment

    process_pool = ProcessPool(N_PROCESSES)
    mp.set_start_method('spawn')

    initial_params_list = [initial_params]
    for hyper_params in itertools.product(ms, ass, der_1st_regs, d_ws, r_ws, t_ws, h_ws, e_ws, momentums, initial_params_list, bound_regs):
        m, a, der_1st_reg, d_w, r_w, t_w, h_w, e_w, momentum, initial_params, bound_reg = hyper_params

        loss_weights = {
            "d_weight": d_w,
            "r_weight": r_w,
            "t_weight": t_w,
            "h_weight": h_w,
            "e_weight": e_w,
        }

        for k, v in loss_weights.items():
            loss_weights[k] = 0.02 * v # 0.0005

        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory=f"runs/FR", uuid_prefix=f"train_size{train_size}_der{der_1st_reg}_m{momentum}_dt_{d_w}")
        process_pool.start(
            target=experiment.run_exp,
            kwargs={
                "initial_params": initial_params,
                "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                "model_params": {"model_cls": TiedSidartheExtended, "der_1st_reg": der_1st_reg, "bound_reg": bound_reg, "bound_loss_type": "log"},
                "train_params": {"momentum": momentum, "m": m, "a": a},
                "loss_weights": loss_weights,
                "plot_references": False
            }
        )
        process_pool.wait_for_empty_slot(timeout=5)

    process_pool.wait_for_all()
