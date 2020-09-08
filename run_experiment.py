import os

from experiments.sidarthe_experiment import SidartheExperiment
from experiments.sidarthe_extended_experiment import ExtendedSidartheExperiment


if __name__ == "__main__":
    region = "Italy"
    n_epochs = 1000
    t_step = 1.0

    experiment_cls = ExtendedSidartheExperiment  # switch class to change experiment: e.g. SidartheExperiment
    experiment = experiment_cls(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs")
    experiment.run_exp()  # params can be set, no params => default configuration

