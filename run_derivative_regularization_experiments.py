import os

from populations import populations

from learning_models.sidarthe_extended import SidartheExtended
from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment

import multiprocessing as mp


experiment_cls = ExtendedSidartheExperiment

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CHOOSE GPU HERE
if __name__ == "__main__":
    region = "Italy"
    t_step = 1.0
    train_size = 120
    val_len = 40

    n_epochs = 1700

    der_1st_regs = [0., 1., 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13]

    procs = []
    mp.set_start_method('spawn')

    for der_1st_reg in der_1st_regs:
        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="derivative_reg", uuid_prefix=None)

        proc = mp.Process(
            target=experiment.run_exp,
            kwargs={
                "initial_params": {},
                "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                "model_params": {"der_1st_reg": der_1st_reg},
                "train_params": {"momentum": False},
            }
        )

        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()