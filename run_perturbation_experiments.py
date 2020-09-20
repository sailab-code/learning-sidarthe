import os
import multiprocessing as mp

from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from torch_euler import Heun

from params import params

N_PROCESSES = 5
N_PERTURBED_RUNS = 10

if __name__ == "__main__":
    region = "Italy"
    t_inc = 1.0
    train_size = 180
    val_len = 10

    momentums = [False, True]
    m,a = 0.125, 0.0
    n_epochs = 10000
    der_1st_reg = 1e5
    bound_reg = 1e6

    loss_weights = {
        "d_weight": 5.,
        "r_weight": 1.,
        "t_weight": 1.,
        "h_weight": 1.,
        "e_weight": 1.
    }

    for k,v in loss_weights.items():
        loss_weights[k] = 0.002 * v


    procs = []
    mp.set_start_method('spawn')
    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment

    for momentum in momentums:
        for r in range(N_PERTURBED_RUNS):
            random_gen = params.SidartheParamGenerator()
            random_gen.random_init(40, ranges="extended")
            print(random_gen.params['gamma'])

            experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_inc, runs_directory=f"initialization_runs/random_perturbation_{r}_m{momentum}")
            proc = mp.Process(
                target=experiment.run_exp,
                kwargs={
                    "initial_params": random_gen.params,
                    "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                    "model_params": {
                        "der_1st_reg": der_1st_reg,
                        "bound_reg": bound_reg,
                        "integrator": Heun
                    },
                    "train_params": {"momentum": momentum, "m": m, "a": a},
                    "loss_weights": loss_weights,
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