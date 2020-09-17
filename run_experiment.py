import os

from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment


if __name__ == "__main__":
    region = "Italy"
    n_epochs = 1000
    t_step = 1.0

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs/test_param_plots")
    experiment.run_exp(dataset_params={
        "train_size": 120,
        "val_len": 40,
        "region": region},
        model_params={
            "der_1st_reg": 1e8,
            "bound_reg": 1.,
            "bound_loss_type": "step"
        },
        train_params={
            "momentum": False
        }
    )  # params can be set, no params => default configuration

