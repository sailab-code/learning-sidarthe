import os

from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment

import multiprocessing as mp
import itertools

N_PROCESSES = 6

if __name__ == "__main__":
    region = "Italy"
    n_epochs = 100
    t_step = 1.0
    train_size = 120
    val_len = 40
    der_1st_regs = [4.1e4]
    t_inc = 1.

    momentums = [False, True]
    ms = [1/8, 1/5, 1/2]
    ass = [0.04, 0.1]
    bound_reg = 1e4
    loss_type = "nrmse"
    d_ws, r_ws, t_ws, h_ws, e_ws = [1.0], [1.0], [1.0], [1.0], [1.0]

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment

    procs = []
    mp.set_start_method('spawn')
    for hyper_params in itertools.product(ms, ass, der_1st_regs, d_ws, r_ws, t_ws, h_ws, e_ws, momentums):
        m, a, der_1st_reg, d_w, r_w, t_w, h_w, e_w, momentum = hyper_params

        loss_weights = {
            "d_weight": d_w,
            "r_weight": r_w,
            "t_weight": t_w,
            "h_weight": h_w,
            "e_weight": e_w,
        }

        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="initialization")

        proc = mp.Process(
            target=experiment.run_exp,
            kwargs={
                "initial_params": {},
                "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                "model_params": {"der_1st_reg": der_1st_reg, "bound_reg": bound_reg},
                "train_params": {"momentum": momentum, "m": m, "a": a},
                "loss_weights": loss_weights
            }
        )
        proc.start()
        procs.append(proc)

        # run N_PROCESSES exps at a time
        if len(procs) == N_PROCESSES:
            for proc in procs:
                proc.join()
            procs.clear()

    for proc in procs:
        proc.join()
