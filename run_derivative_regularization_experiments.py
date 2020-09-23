import os

from populations import populations

from learning_models.sidarthe_extended import SidartheExtended
from learning_models.tied_sidarthe_extended import TiedSidartheExtended
from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment
from utils.multiprocess_utils import ProcessPool
from params.params import SidartheParamGenerator

import multiprocessing as mp


experiment_cls = ExtendedSidartheExperiment

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CHOOSE GPU HERE

N_PROCESSES = 32

if __name__ == "__main__":
    region = "Italy"
    t_step = 1.0
    train_size = 120
    val_len = 40

    n_epochs = 2000

    der_1st_regs = [0., 1., 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13]
    n_tries = 20

    process_pool = ProcessPool(N_PROCESSES)
    mp.set_start_method('spawn')

    for n_try in range(0, n_tries):

        gen = SidartheParamGenerator()
        gen.random_init(train_size-5, ranges="extended")

        for der_1st_reg in der_1st_regs:
            experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs/derivative_reg", uuid_prefix=None)

            process_pool.start(
                target=experiment.run_exp,
                kwargs={
                    "initial_params": gen.params,
                    "dataset_params": {"train_size": train_size, "val_len": val_len, "region": region},
                    "model_params": {"der_1st_reg": der_1st_reg, "model_cls": TiedSidartheExtended, "bound_loss_type": "step"},
                    "train_params": {"momentum": False},
                }
            )

            process_pool.wait_for_empty_slot()

    process_pool.wait_for_all()