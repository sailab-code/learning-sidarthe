import os

from lcm.datasets.st_sidarthe_dataset import SpatioTemporalSidartheDataset
from lcm.trainers.sidarthe_extended_trainer import SidartheExtendedTrainer
from lcm.integrators.fixed_step import Heun
from lcm.losses import compose_losses
from lcm.losses.regularization_losses import LteZero, FirstDerivative
from lcm.losses.target_losses import RMSE
from lcm.st_sidarthe import SpatioTemporalSidarthe
from lcm.utils.populations import populations


class SpatioTemporalSidartheTrainer(SidartheExtendedTrainer):
    def make_initial_params(self, **kwargs):
        # default initial_params
        initial_params = {
            "alpha": [[0.03]]*100,
            "beta": [[0.01]]*100,
            "gamma": [[0.12]]*100,
            "delta": [[0.01]]*100,
            "epsilon": [[0.14]]*100,
            "theta": [[0.371]]*100,
            "zeta": [[0.08]]*100,
            "eta": [[0.08]]*100,
            "mu": [[0.01]]*100,
            "nu": [[0.01]]*100,
            "tau": [[0.01]]*100,
            "lambda": [[0.01]]*100,
            "kappa": [[0.01]]*100,
            "xi": [[0.01]]*100,
            "rho": [[0.01]]*100,
            "sigma": [[0.01]]*100,
            "phi": [[0.01]]*100,
            # "chi": [[0.01, 0.03, 0.014]]*100,
            "chi": [[0.01]]*100,
        }  # fixme is temporary


        _params = kwargs.get("initial_params", {})

        return self.fill_missing_params(_params, initial_params)

    def make_model(self, **kwargs):
        return self.model_params["model_cls"](
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
        model_cls = SpatioTemporalSidarthe
        ppls = [populations[area] for area in self.dataset.region]

        model_params = {
            "model_cls": model_cls,
            "population": ppls, # tensor of size S
            "initial_conditions": self.dataset.get_initial_conditions(ppls), # S x 8
            # "initial_conditions": [(59999576.0, 94, 94, 101, 101, 26, 7, 1)]*3, # S x 8
            "integrator": Heun,
            "n_areas": self.dataset.n_areas,
            "time_step": self.time_step,
            "loss_fn": RMSE({
                "d": 0.02,
                "r": 0.02,
                "t": 0.02,
                "h": 0.02,
                "e": 0.02,
            }),
            "reg_fn": compose_losses(
                [
                    LteZero(1e6),
                    FirstDerivative(1e6, self.time_step)
                ]
            )
        }

        _params = kwargs.get("model_params", {})

        return self.fill_missing_params(_params, model_params)
