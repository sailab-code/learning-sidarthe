import os
import multiprocessing as mp

from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from utils.params_initializers import perturb, constant_perturb
from torch_euler import Heun

N_PROCESSES = 5
N_PERTURBED_RUNS = 10

if __name__ == "__main__":
    exps_dir = os.path.join(os.getcwd(), "nature_init", "sidarthe_extended", "Italy")  # Robustness exps directory
    # exps_list = ["9e4337ca-554a-4bc9-bfd7-513e40cbac46"] # list of exps to evaluate, we may also get them all
    exps_list = ["620ae948-e78d-441f-b119-fe60505ddab8"] # list of exps to evaluate, we may also get them all

    n_epochs = 12000
    val_len = 10

    procs = []
    mp.set_start_method('spawn')
    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment



    for exp in exps_list:
        exp_path = exp_dir = os.path.join(exps_dir, exp)
        # initialization_params = experiment_cls.get_configs_from_json(os.path.join(exp_path, "final.json"))["params"]
        settings = experiment_cls.get_configs_from_json(os.path.join(exp_path, "settings.json"))

        for r in range(N_PERTURBED_RUNS):
            mu, sigma = 0.0, 0.01
            perturbed_params = perturb(settings["initial_values"], mu=mu, sigma=sigma, seed=r)
            # perturbed_params = constant_perturb(settings["initial_values"], mu=mu, sigma=sigma, seed=r)

            experiment = experiment_cls(settings["region"], n_epochs=n_epochs, time_step=settings["t_inc"], runs_directory=f"with_momentum_perturbed_{exp}")
            print(settings["der_1st_reg"])
            proc = mp.Process(
                target=experiment.run_exp,
                kwargs={
                    "initial_params": perturbed_params,
                    "dataset_params": {"train_size": settings["train_size"], "val_len": val_len, "region": settings["region"]},
                    "model_params": {
                        "der_1st_reg": 5e3*settings["der_1st_reg"],
                        "bound_reg": settings["bound_reg"],
                        "loss_type": settings["loss_type"],
                        "integrator": Heun
                    },
                    "train_params": {"momentum": True, "m": 0.125, "a": 0.1},
                    "loss_weights": settings["target_weights"],
                    "learning_rates": settings["learning_rates"]
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
