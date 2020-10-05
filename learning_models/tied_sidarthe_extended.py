from typing import Dict, List

import torch
from torch.optim import Optimizer

from learning_models.sidarthe_extended import SidartheExtended
from learning_models.optimizers.tied_optimizer import NewSirOptimizer
# from learning_models.optimizers.tied_optimizer import TiedOptimizer


class TiedSidartheExtended(SidartheExtended):
    def __init__(self, parameters: Dict, population, init_cond, integrator, sample_time, **kwargs):
        super().__init__(parameters, population, init_cond, integrator, sample_time, **kwargs)
        self.model_name = kwargs.get("name", "tied_sidarthe_extended")

    @property
    def params(self) -> Dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "theta": self.theta,
            "xi": self.xi,
            "eta": self.eta,
            "mu": self.mu,
            "nu": self.nu,
            "tau": self.tau,
            "lambda": self.lambda_,
            "kappa": self.kappa,
            "zeta": self.zeta,
            "rho": self.rho,
            "sigma": self.sigma,
            "phi": self.phi,
            "chi": self.chi
        }

    @property
    def alpha(self) -> torch.Tensor:
        return self._params["alpha"]

    @property
    def beta(self) -> torch.Tensor:
        return self._params["beta"]

    @property
    def gamma(self) -> torch.Tensor:
        return self._params["gamma"]

    # tied to beta
    @property
    def delta(self) -> torch.Tensor:
        return self._params["beta"]

    @property
    def epsilon(self) -> torch.Tensor:
        return self._params["epsilon"]

    @property
    def theta(self) -> torch.Tensor:
        return self._params["theta"]

    @property
    def xi(self) -> torch.Tensor:
        return self._params["xi"]

    @property
    def eta(self) -> torch.Tensor:
        return self._params["eta"]

    @property
    def mu(self) -> torch.Tensor:
        return self._params["mu"]

    @property
    def nu(self) -> torch.Tensor:
        return self._params["nu"]

    @property
    def tau(self) -> torch.Tensor:
        return self._params["tau"]

    # tied to rho
    @property
    def lambda_(self) -> torch.Tensor:
        return self._params["rho"]

    @property
    def kappa(self) -> torch.Tensor:
        return self._params["xi"]

    # tied to eta
    @property
    def zeta(self) -> torch.Tensor:
        return self._params["eta"]

    @property
    def rho(self) -> torch.Tensor:
        return self._params["rho"]

    @property
    def sigma(self) -> torch.Tensor:
        return self._params["sigma"]

    @property
    def phi(self) -> torch.Tensor:
        return self._params["phi"]

    @property
    def chi(self) -> torch.Tensor:
        return self._params["chi"]

    @property
    def tied_params(self) -> Dict:
        return {
            "delta": "beta",
            "lambda": "rho",
            "zeta": "eta"
        }

    @classmethod
    def init_optimizers(cls, model: 'TiedSidartheExtended', learning_rates: dict, optimizers_params: dict) -> List[Optimizer]:
        m = optimizers_params.get("m", 1 / 9)
        a = optimizers_params.get("a", 0.05)
        momentum = optimizers_params.get("momentum", True)
        summary = optimizers_params.get("tensorboard_summary", None)

        optimizer = NewSirOptimizer(model._params, learning_rates, m=m, a=a, momentum=momentum, summary=summary)
        return [optimizer]