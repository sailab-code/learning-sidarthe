import os
import pandas as pd

from torch_euler import Heun
from populations import populations
from learning_models.sidarthe_extended import SidartheExtended
from experiments.sidarthe_experiment import SidartheExperiment


class ExtendedSidartheExperiment(SidartheExperiment):
    """
    Class to run experiments of Extended Sidarthe. In principle this class is not
    needed to run experiments using ExtendedSidarthe model, however it provides already
    defaults configuration for the extended sidarthe version, runnable as follows:
        experiment = ExtendedSidartheExperiment(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs")
        experiment.run_exp() # without any input param

    """

    def __init__(self, region, n_epochs, time_step, runs_directory="runs"):
        super().__init__(
            region,
            n_epochs,
            time_step,
            runs_directory
        )

    def make_initial_params(self, **kwargs):
        # default initial_params
        initial_params = {
            "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * 64,
            "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 63),
            "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 10 + [0.11] * 64,
            "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 63),
            "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * 64,
            "theta": [0.371],
            "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * 64,
            "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * 64,
            "mu": [0.017] * 22 + [0.008] * (17 + 63),
            "nu": [0.027] * 22 + [0.015] * (17 + 63),
            "tau": [0.15] * 102,
            "lambda": [0.034] * 22 + [0.08] * (17 + 63),
            "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * 64,
            "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * 64,
            "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * 64,
            "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * 64,
            "phi": [0.02] * 102,
            "chi": [0.02] * 102
        }
        return kwargs.get("initial_params", initial_params)

    def make_model_params(self, **kwargs):
        # default model params
        model_params = {
            "model_cls": SidartheExtended,
            "name": "sidarthe_extended",
            "der_1st_reg": 50000.0,
            "population": populations[self.region],
            "integrator": Heun,
            "time_step": self.time_step,
            "bound_reg": 1e5,
            "loss_type": "nrmse",
            "verbose": False,
            "val_size": self.dataset.val_len,
            "train_size": self.dataset.train_size,
            "references": self.references,
            "first_date": self.dataset.first_date,
        }
        return kwargs.get("model_params", model_params)

    def make_learning_rates(self, **kwargs):
        # default learning rates
        learning_rates = {
            "alpha": 1e-5,
            "beta": 1e-6,
            "gamma": 1e-5,
            "delta": 1e-6,
            "epsilon": 1e-5,
            "theta": 1e-7,
            "xi": 1e-5,
            "eta": 1e-5,
            "mu": 1e-5,
            "nu": 1e-5,
            "tau": 1e-5,
            "lambda": 1e-5,
            "kappa": 1e-5,
            "zeta": 1e-5,
            "rho": 1e-5,
            "sigma": 1e-5,
            "phi": 1e-5,
            "chi": 1e-5
        }
        return kwargs.get("learning_rates", learning_rates)


