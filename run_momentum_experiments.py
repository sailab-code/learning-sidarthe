import os

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

N_PROCESSES = 32
experiment_cls = ExtendedSidartheExperiment

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CHOOSE GPU HERE
if __name__ == "__main__":
    region = "Italy"
    t_step = 1.0
    train_size = 120
    val_len = 40

    der_1st_reg = 1e8

    n_epochs = 3000

    process_pool = ProcessPool(N_PROCESSES)
    mp.set_start_method('spawn')

    m_space = np.linspace(0., 0.5, 6)
    a_space = np.linspace(0., 0.2, 5)
    n_tries = 20

    bound_reg = 1e0

    kwargs = {
        "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
        "model_params": {
            "der_1st_reg": der_1st_reg, 
            "model_cls": TiedSidartheExtended, 
            "bound_loss_type": "log", 
            "bound_reg": bound_reg,
            "loss_type": "nrmse"
        }
    }

    runs_directory = "runs/momentum_exps"

    for n_try in range(0, n_tries):
        gen = SidartheParamGenerator()    
        gen.set_param_types(param_types={'tau': 'dyn', 'theta': 'dyn'})
        gen.random_init(115, ranges="extended")

        
        experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory=runs_directory, uuid_prefix=None)
        # launch experiment with momentum=False
        process_pool.start(
                target=experiment.run_exp,
                kwargs={
                    **kwargs,
                    "initial_params": gen.params,
                    "train_params": {"momentum": False},
                }
            )

        process_pool.wait_for_empty_slot()

        # launch experiments with momentum=True
        for m, a in itertools.product(m_space, a_space):
            experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory=runs_directory, uuid_prefix=None)
            process_pool.start(
                target=experiment.run_exp,
                kwargs={
                    **kwargs,
                    "train_params": {"momentum": True, "m": m, "a": a},
                }
            )
            process_pool.wait_for_empty_slot()

    process_pool.wait_for_all()