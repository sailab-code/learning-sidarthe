import os

from lcm.datasets.st_sidarthe_dataset import SpatioTemporalSidartheDataset
from lcm.experiments.sidarthe_extended_experiment import SidartheExtendedExperiment
from lcm.integrators.fixed import Heun
from lcm.losses import compose_losses
from lcm.losses.regularization_losses import LteZero, FirstDerivative
from lcm.losses.target_losses import RMSE
from lcm.sidarthe import Sidarthe
from lcm.utils.populations import populations


class SidartheExperiment(SidartheExtendedExperiment):
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

    def make_model(self, **kwargs):
        return Sidarthe(
            **self.model_params,
            params=self.initial_params,
            learning_rates=self.learning_rates,
            momentum_settings=self.train_params['momentum_settings']
        )

    def make_dataset_params(self, **kwargs):
        # default dataset params
        dataset_params = {
            "dataset_cls": SpatioTemporalSidartheDataset,
            "region": self.region,
            "data_path": os.path.join(os.getcwd(), "data", "COVID-19", "dati-andamento-nazionale", "dpc-covid19-ita-andamento-nazionale.csv"),
            "train_size": 110,
            "val_size": 5,
        }
        _params = kwargs.get("dataset_params", {})

        return self.fill_missing_params(_params, dataset_params)

    def make_model_params(self, **kwargs):
        # default pretrained_model params
        model_params = {
            "population": populations[self.region],
            "initial_conditions": (59999576.0, 94, 94, 101, 101, 26, 7, 1), #fixme
            "integrator": Heun,
            "n_areas": self.dataset.n_areas,
            "time_step": self.time_step,
            "loss_fn": RMSE({
                "d": 1.,
                "r": 1.,
                "t": 1.,
                "h": 1.,
                "e": 1.,
            }),
            "reg_fn": compose_losses(
                [
                    LteZero(1.),
                    FirstDerivative(1., self.time_step)
                ]
            )
        }

        _params = kwargs.get("model_params", {})

        return self.fill_missing_params(_params, model_params)
