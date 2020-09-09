import os
import multiprocessing as mp

from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from utils.params_initializers import perturb

N_PROCESSES = 6
N_PERTURBED_RUNS = 10

if __name__ == "__main__":
    exps_dir = os.path.join(os.getcwd(), "initialization")  # Robustness exps directory
    exps_list = [] # TODO add the list of exps to evaluate, we may also get them all

    n_epochs = 8000
    train_size = 120
    val_len = 40

    procs = []
    mp.set_start_method('spawn')
    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    for exp in exps_list:
        exp_path = exp_dir = os.path.join(exps_dir, exp)
        # initialization_params = experiment_cls.get_configs_from_json(os.path.join(exp_path, "final.json"))["params"]
        settings = experiment_cls.get_configs_from_json(os.path.join(exp_path, "settings.json"))

        mu, sigma = 0.0, 0.1
        perturbed_params = perturb(settings["initial_values"], mu=mu, sigma=sigma)

        for r in range(N_PERTURBED_RUNS):
            experiment = experiment_cls(settings["region"], n_epochs=n_epochs, time_step=settings["t_inc"], runs_directory=f"perturbed_{exp}")
            proc = mp.Process(
                target=experiment.run_exp,
                kwargs={
                    "initial_params": perturbed_params,
                    "dataset_params": {"train_size": train_size, "val_len": val_len, "region": settings["region"]},
                    "model_params": {"der_1st_reg": settings["der_1st_reg"],
                                     "bound_reg": settings["bound_reg"],
                                     "loss_type": settings["loss_type"],
                                     "integrator": settings["integrator"]
                                     },
                    "train_params": {"momentum": settings["momentum"], "m": settings["m"], "a": settings["a"]},
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
