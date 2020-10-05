import os
import random

from params.params import SidartheParamGenerator
from populations import populations

from learning_models.sidarthe_extended import SidartheExtended
from learning_models.tied_sidarthe_extended import TiedSidartheExtended

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

    n_epochs = 3000

    procs = []
    process_pool = ProcessPool(N_PROCESSES)
    mp.set_start_method('spawn')


    bound_reg = 1e0
    bound_loss_type = "log"
    loss_type = "nrmse"

    kwargs = {
        "model_params": {
            "der_1st_reg": der_1st_reg,
            "model_cls": TiedSidartheExtended,
            "bound_loss_type": bound_loss_type,
            "bound_reg": bound_reg,
            "loss_type": loss_type
        }    
    }

    m_space = np.linspace(0., 0.5, 6)
    train_sizes = range(40, 121, 20)
    n_tries = 20

    runs_directory = "runs/momentum_train_size_exps"

    for n_try in range(0, n_tries):
        gen = SidartheParamGenerator()
        gen.set_param_types(param_types={'tau': 'dyn', 'theta': 'dyn'})
        gen.random_init(40, ranges="extended")
        
        for train_size in train_sizes:
            gen.extend(train_size)
            
            # launch experiment with momentum=False
            experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step,
                                        runs_directory=runs_directory, uuid_prefix=None)
            process_pool.start(target=experiment.run_exp,
                               kwargs={
                                   **kwargs,
                                   "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                                   "initial_params": gen.params,
                                   "train_params": {"momentum": False},
                               })

            process_pool.wait_for_empty_slot(timeout=5)

            for m in m_space:
                experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step,
                                        runs_directory=runs_directory, uuid_prefix=None)
                process_pool.start(target=experiment.run_exp,
                               kwargs={
                                   **kwargs,
                                   "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                                   "initial_params": gen.params,
                                   "train_params": {"momentum": True, "m": m, "a": 0.},
                               })
                               
                process_pool.wait_for_empty_slot(timeout=5)


    process_pool.wait_for_all()