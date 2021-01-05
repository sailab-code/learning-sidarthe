from lcm.trainers.st_sidarthe_trainer import SpatioTemporalSidartheTrainer
from lcm.integrators.fixed_step import Heun
from lcm.losses import compose_losses
from lcm.losses.regularization_losses import LteZero, FirstDerivative
from lcm.losses.target_losses import RMSE
from lcm.st_sidarthe import SpatioTemporalSidarthe
from lcm.utils.populations import populations


class MobilitySpatioTemporalSidartheTrainer(SpatioTemporalSidartheTrainer):
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
            "mobility_0": [[2.]]
        }  # fixme is temporary


        _params = kwargs.get("initial_params", {})

        return self.fill_missing_params(_params, initial_params)

    def make_model_params(self, **kwargs):
        # default pretrained_model params
        model_cls = SpatioTemporalSidarthe
        ppls = [populations[area] for area in self.dataset.region]

        model_params = {
            "model_cls": model_cls,
            "population": ppls, # tensor of size S
            "mobility": self.dataset.get_mobility(),
            "initial_conditions": self.dataset.get_initial_conditions(ppls), # S x 8
            "integrator": Heun,
            "n_areas": self.dataset.n_areas,
            "time_step": self.time_step,
            "loss_fn": RMSE({
                "d": 0.03,
                "r": 0.02,
                "t": 0.02,
                "h": 0.02,
                "e": 0.03,
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
