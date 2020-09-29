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
import json

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

    params_path = os.path.join(os.getcwd(), "runs", "momentum_exps", "sidarthe_extended", "Italy", "initial_params.json")

    with open(params_path) as f:
        params_list = json.load(f)

    for initial_params in params_list:
        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs/momentum_False_exps", uuid_prefix=None)

        process_pool.start(
            target=experiment.run_exp,
            kwargs={
                "initial_params": initial_params,
                "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                "model_params": {"der_1st_reg": der_1st_reg, "bound_loss_type": "step"},
                "train_params": {"momentum": False},
            }
        )

        process_pool.wait_for_empty_slot()

    process_pool.wait_for_all()