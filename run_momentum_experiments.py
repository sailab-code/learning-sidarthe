import os

from params.params import SidartheParamGenerator
from populations import populations

from learning_models.sidarthe_extended import SidartheExtended
from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment

import multiprocessing as mp
import numpy as np
import itertools

from utils.multiprocess_utils import ProcessPool

N_PROCESSES = 10
experiment_cls = ExtendedSidartheExperiment

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CHOOSE GPU HERE
if __name__ == "__main__":
    region = "Italy"
    t_step = 1.0
    train_size = 120
    val_len = 40

    der_1st_reg = 1e8

    n_epochs = 2000

    process_pool = ProcessPool()
    mp.set_start_method('spawn')

    m_space = np.linspace(0., 0.5, 6)
    a_space = np.linspace(0., 0.2, 6) * -1
    n_tries = 1

    for n_try in range(0, n_tries):
        gen = SidartheParamGenerator()
        gen.random_init(115, ranges="extended")
        for m, a in itertools.product(m_space, a_space):
            experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs/momentum_exps", uuid_prefix=None)

            process_pool.start(
                target=experiment.run_exp,
                kwargs={
                    "initial_params": gen.params,
                    "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                    "model_params": {"der_1st_reg": der_1st_reg},
                    "train_params": {"momentum": True, "m": m, "a": a},
                }
            )

            process_pool.wait_for_empty_slot()

    process_pool.wait_for_all()