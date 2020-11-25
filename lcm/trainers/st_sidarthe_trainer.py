import os

from lcm.datasets.st_sidarthe_dataset import SpatioTemporalSidartheDataset
from lcm.trainers.sidarthe_extended_trainer import SidartheExtendedExperiment
from lcm.integrators.fixed import Heun
from lcm.losses import compose_losses
from lcm.losses.regularization_losses import LteZero, FirstDerivative
from lcm.losses.target_losses import RMSE
from lcm.st_sidarthe import SpatioTemporalSidarthe
# from lcm.utils.populations import populations


class SpatioTemporalSidartheTrainer(SidartheExtendedExperiment):
    def make_initial_params(self, **kwargs):
        # default initial_params
        initial_params = {
            "alpha": [[0.03]],
            "beta": [[0.01]],
            "gamma": [[0.12]],
            "delta": [[0.01]],
            "epsilon": [[0.14]],
            "theta": [[0.371]],
            "zeta": [[0.08]],
            "eta": [[0.08]],
            "mu": [[0.01]],
            "nu": [[0.01]],
            "tau": [[0.01]],
            "lambda": [[0.01]],
            "kappa": [[0.01]],
            "xi": [[0.01]],
            "rho": [[0.01]],
            "sigma": [[0.01]],
            "phi": [[0.01]],
            "chi": [[0.01]],
        }  # fixme is temporary


        _params = kwargs.get("initial_params", {})

        return self.fill_missing_params(_params, initial_params)

    def make_model(self, **kwargs):
        return SpatioTemporalSidarthe(
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
            "region_column": "denominazione_regione"
        }
        _params = kwargs.get("dataset_params", {})

        return self.fill_missing_params(_params, dataset_params)

    def make_model_params(self, **kwargs):
        # default pretrained_model params
        model_params = {
            "population": 59999576.0, # populations[self.region], fixme
            "initial_conditions": [(59999576.0, 94, 94, 101, 101, 26, 7, 1)]*3,
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
