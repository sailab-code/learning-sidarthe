import os
import pandas as pd

from torch_euler import Heun
from populations import populations
from dataset.sidarthe_dataset import SidartheDataset
from learning_models.sidarthe import Sidarthe
from experiments.experiment import Experiment


class SidartheExperiment(Experiment):
    def __init__(self, region, n_epochs, time_step, runs_directory="runs", uuid=None):
        super().__init__(
            region,
            n_epochs,
            time_step,
            runs_directory,
            uuid
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
        }
        _params = kwargs.get("initial_params", {})

        return self.fill_missing_params(_params, initial_params)

    def make_dataset_params(self, **kwargs):
        # default dataset params
        dataset_params = {
            "dataset_cls": SidartheDataset,
            "region": self.region,
            "train_size": 110,
            "val_len": 5,
        }
        _params = kwargs.get("dataset_params", {})

        return self.fill_missing_params(_params, dataset_params)

    def make_model_params(self, **kwargs):
        # default pretrained_model params
        model_params = {
            "model_cls": Sidarthe,
            "name": "sidarthe",
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
        _params = kwargs.get("model_params", {})

        return self.fill_missing_params(_params, model_params)

    def make_train_params(self, **kwargs):
        # default train params
        train_params = {
            "t_start": 0,
            "t_end": self.dataset.train_size,
            "val_size": self.dataset.val_len,
            "time_step": self.time_step,
            "m": 0.125,
            "a": 0.1,
            "momentum": True,
        }
        _params = kwargs.get("train_params", {})

        return self.fill_missing_params(_params, train_params)

    def make_loss_weights(self, **kwargs):
        # default loss weights
        loss_weights = {
            "d_weight": 1.,
            "r_weight": 1.,
            "t_weight": 1.,
            "h_weight": 1.,
            "e_weight": 1.
        }
        _params = kwargs.get("loss_weights", {})

        return self.fill_missing_params(_params, loss_weights)

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
            "sigma": 1e-5
        }
        _params = kwargs.get("learning_rates", {})

        return self.fill_missing_params(_params, learning_rates)

    def make_references(self):
        references = {}
        ref_df = pd.read_csv(os.path.join(os.getcwd(), "regioni/sidarthe_results_new.csv"))
        for key in 'sidarthe':
            references[key] = ref_df[key].tolist()

        for key in ["r0", "h_detected"]:
            references[key] = ref_df[key].tolist()

        for key in self.initial_params.keys():
            if key in ref_df:
                references[key] = ref_df[key].tolist()

        return references
