import os
import pandas as pd

from torch_euler import Heun
from populations import populations
from learning_models.sidarthe_extended import SidartheExtended
from experiments.sidarthe_experiment import SidartheExperiment


class SidartheExtendedExperiment(SidartheExperiment):
    """
    Class to run experiments of Extended Sidarthe. In principle this class is not
    needed to run experiments using ExtendedSidarthe pretrained_model, however it provides already
    defaults configuration for the extended sidarthe version, runnable as follows:
        experiment = ExtendedSidartheExperiment(region, n_epochs=n_epochs, time_step=t_step, runs_directory="runs")
        experiment.run_exp() # without any input param

    """

    def __init__(self, region, n_epochs, time_step, runs_directory="runs", uuid=None, uuid_prefix=None):
        super().__init__(
            region,
            n_epochs,
            time_step,
            runs_directory,
            uuid,
            uuid_prefix
        )

    def make_initial_params(self, **kwargs):
        # default initial_params
        initial_params = {
            "alpha": [0.570] * 4 + [0.422] * 18 + [0.360] * 6 + [0.210] * 10 + [0.210] * 142,
            "beta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 141),
            "gamma": [0.456] * 4 + [0.285] * 18 + [0.2] * 6 + [0.11] * 10 + [0.11] * 142,
            "delta": [0.011] * 4 + [0.0057] * 18 + [0.005] * (17 + 141),
            "epsilon": [0.171] * 12 + [0.143] * 26 + [0.2] * 142,
            "theta": [0.371],
            "zeta": [0.125] * 22 + [0.034] * 16 + [0.025] * 142,
            "eta": [0.125] * 22 + [0.034] * 16 + [0.025] * 142,
            "mu": [0.017] * 22 + [0.008] * (17 + 141),
            "nu": [0.027] * 22 + [0.015] * (17 + 141),
            "tau": [0.15],
            "lambda": [0.034] * 22 + [0.08] * (17 + 141),
            "kappa": [0.017] * 22 + [0.017] * 16 + [0.02] * 142,
            "xi": [0.017] * 22 + [0.017] * 16 + [0.02] * 142,
            "rho": [0.034] * 22 + [0.017] * 16 + [0.02] * 142,
            "sigma": [0.017] * 22 + [0.017] * 16 + [0.01] * 142,
            "phi": [0.02] * 180,
            "chi": [0.02] * 180
        }
        _params = kwargs.get("initial_params", {})

        return self.fill_missing_params(_params, initial_params)

    def make_model_params(self, **kwargs):
        # default pretrained_model params
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
            "n_areas": self.dataset.batch_size,
            "val_size": self.dataset.val_len,
            "train_size": self.dataset.train_size,
            "references": self.references,
            "first_date": self.dataset.first_date,
        }
        _params = kwargs.get("model_params", {})

        return self.fill_missing_params(_params, model_params)

    def make_learning_rates(self, **kwargs):
        # default learning rates
        learning_rates = super().make_learning_rates(**kwargs)
        learning_rates["phi"] = 1e-5
        learning_rates["chi"] = 1e-5

        _params = kwargs.get("learning_rates", {})

        return self.fill_missing_params(_params, learning_rates)
