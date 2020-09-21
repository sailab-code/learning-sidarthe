import os
import random

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

    procs = []
    process_pool = ProcessPool(N_PROCESSES)
    mp.set_start_method('spawn')

    m_space = np.linspace(0., 0.5, 11)
    train_sizes = range(40, 120, 20)
    n_tries = 3

    for n_try in range(0, n_tries):
        gen = SidartheParamGenerator()
        gen.random_init(40, ranges="extended")
        for train_size, m in itertools.product(train_sizes, m_space):
            experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step,
                                        runs_directory="runs/momentum_train_size_exps", uuid_prefix=None)
            gen.extend(train_size)

            process_pool.start(target=experiment.run_exp,
                               kwargs={
                                   "initial_params": gen.params,
                                   "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                                   "model_params": {"der_1st_reg": der_1st_reg},
                                   "train_params": {"momentum": True, "m": m, "a": 0.},
                               })

            process_pool.wait_for_empty_slot(timeout=5)

    process_pool.wait_for_all()