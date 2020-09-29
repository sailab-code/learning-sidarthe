import os

from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment

from learning_models.tied_sidarthe_extended import TiedSidartheExtended
from params.params import SidartheParamGenerator
from utils.multiprocess_utils import ProcessPool

import multiprocessing as mp
import itertools

N_PROCESSES = 6

if __name__ == "__main__":
    region = "Italy"
    n_epochs = 5000
    t_step = 1.0
    train_size = 187
    val_len = 10
    der_1st_regs = [1e8]
    t_inc = 1.

    momentums = [True]
    ms = [0.01]
    ass = [0.]
    bound_regs = [1e0]
    loss_type = "nrmse"
    bound_loss_type = "log"
    d_ws, r_ws, t_ws, h_ws, e_ws = [4.], [2.0], [1.0], [2.0], [1.0]

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment


    gen = SidartheParamGenerator()
    gen.init_from_base_params("extended")
    gen.set_param_types(param_types={'tau': 'dyn', 'theta': 'dyn'})
    gen.extend(train_size-5)
    process_pool = ProcessPool(N_PROCESSES)


    procs = []
    mp.set_start_method('spawn')
    for hyper_params in itertools.product(ms, ass, der_1st_regs, d_ws, r_ws, t_ws, h_ws, e_ws, momentums, bound_regs):
        m, a, der_1st_reg, d_w, r_w, t_w, h_w, e_w, momentum, bound_reg = hyper_params

        loss_weights = {
            "d_weight": d_w,
            "r_weight": r_w,
            "t_weight": t_w,
            "h_weight": h_w,
            "e_weight": e_w,
        }

        for k,v in loss_weights.items():
            loss_weights[k] = 0.02 * v

        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs/iaml_exps/tied_opt", uuid_prefix=f"{bound_reg:.0e}")

        proc = mp.Process(
            target=experiment.run_exp,
            kwargs={
                "initial_params": gen.params,
                "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                "model_params": {
                    "der_1st_reg": der_1st_reg, 
                    "bound_reg": bound_reg, 
                    "model_cls": TiedSidartheExtended,
                    "bound_loss_type": "log",
                    "loss_type": loss_type
                    },
                "train_params": {"momentum": momentum, "m": m, "a": a, "log_epoch_steps": 50},
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
