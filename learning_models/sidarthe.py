from collections import namedtuple
from typing import List, Dict

import numpy
import torch
from torch.optim.optimizer import Optimizer

from learning_models.abstract_model import AbstractModel
from learning_models.new_sir_optimizer import NewSirOptimizer
from learning_models.sir_optimizer import SirOptimizer
from utils import derivatives

Parameters = namedtuple("Parameters", (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "csi",
    "eta",
    "mu",
    "ni",
    "tau",
    "lamba",
    "kappa",
    "zeta",
    "rho",
    "sigma"
))
Parameters.__new__.__defaults__ = (0.,) * len(Parameters._fields)
# {key: torch.tensor(value, dtype=self.dtype, requires_grad=True) for key, value in parameters._asdict().items()}

class Sidharte(AbstractModel):
    dtype = torch.float64

    def __init__(self, parameters: Dict, population, init_cond, integrator, sample_time, **kwargs):
        super().__init__(init_cond, integrator, sample_time)
        self._params = parameters
        self.population = population

        self.d_weight = kwargs["d_weight"]
        self.r_weight = kwargs["r_weight"]
        self.t_weight = kwargs["t_weight"]
        self.h_weight = kwargs["h_weight"]
        self.e_weight = kwargs["e_weight"]
        self.der_1st_reg = kwargs["der_1st_reg"]
        self.bound_reg = kwargs["bound_reg"]


    @property
    def params(self) -> Dict:
        return self._params

    #region ModelParams

    @property
    def alpha(self) -> torch.Tensor:
        return self._params["alpha"]

    @property
    def beta(self) -> torch.Tensor:
        return self._params["beta"]

    @property
    def gamma(self) -> torch.Tensor:
        return self._params["gamma"]

    @property
    def delta(self) -> torch.Tensor:
        return self._params["delta"]

    @property
    def epsilon(self) -> torch.Tensor:
        return self._params["epsilon"]

    @property
    def theta(self) -> torch.Tensor:
        return self._params["theta"]

    @property
    def csi(self) -> torch.Tensor:
        return self._params["csi"]

    @property
    def eta(self) -> torch.Tensor:
        return self._params["eta"]

    @property
    def mu(self) -> torch.Tensor:
        return self._params["mu"]

    @property
    def ni(self) -> torch.Tensor:
        return self._params["ni"]

    @property
    def tau(self) -> torch.Tensor:
        return self._params["tau"]

    @property
    def lambda_(self) -> torch.Tensor:
        return self._params["lambda"]

    @property
    def kappa(self) -> torch.Tensor:
        return self._params["kappa"]

    @property
    def zeta(self) -> torch.Tensor:
        return self._params["zeta"]

    @property
    def rho(self) -> torch.Tensor:
        return self._params["rho"]

    @property
    def sigma(self) -> torch.Tensor:
        return self._params["sigma"]

    #endregion CodeParams

    def differential_equations(self, t, x):
        """
        Returns the right-hand side of SIDARTHE model
        :param t: time t at which right-hand side is computed
        :param x: state of model at time t
            x[0] = S
            x[1] = I
            x[2] = D
            x[3] = A
            x[4] = R
            x[5] = T
            x[6] = H
            x[7] = E
        :return: right-hand side of SIDARTHE model, i.e. f(t,x(t))
        """

        def get_param_at_t(param, _t):
            if 0 < _t < param.shape[0]:
                return param[_t]
            else:
                return param[-1]

        #region parameters
        alpha = get_param_at_t(self.alpha, t) / self.population
        beta = get_param_at_t(self.beta, t) / self.population
        gamma = get_param_at_t(self.gamma, t) / self.population
        delta = get_param_at_t(self.delta, t) / self.population

        epsilon = get_param_at_t(self.epsilon, t)
        theta = get_param_at_t(self.theta, t)
        csi = get_param_at_t(self.csi, t)
        eta = get_param_at_t(self.eta, t)
        mu = get_param_at_t(self.mu, t)
        ni = get_param_at_t(self.alpha, t)
        tau = get_param_at_t(self.tau, t)
        lambda_ = get_param_at_t(self.lambda_, t)
        kappa = get_param_at_t(self.kappa, t)
        zeta = get_param_at_t(self.zeta, t)
        rho = get_param_at_t(self.rho, t)
        sigma = get_param_at_t(self.sigma, t)
        #endregion parameters

        S = x[0]
        I = x[1]
        D = x[2]
        A = x[3]
        R = x[4]
        T = x[5]
        # H = x[6]
        # E = x[7]

        #region equations

        S_dot = -S * (alpha * I + beta * D + gamma * A + delta * R)
        I_dot = -S_dot - (epsilon + csi + lambda_) * I
        D_dot = epsilon * I - (eta + rho) * D
        A_dot = csi * I - (theta + mu + kappa) * A
        R_dot = eta * D + theta * A - (ni + zeta) * R
        T_dot = mu * A + ni * R - (sigma + tau) * T
        # H_dot = lambda_ * I + rho * D + kappa * A + zeta * R + sigma * T
        # E_dot = tau * T

        #endregion equations

        return torch.cat((
            S_dot,
            I_dot,
            D_dot,
            A_dot,
            R_dot,
            T_dot,
            # H_dot,
            # E_dot
        ), dim=0)

    def omega(self, t):
        if t >= 0:
            return self.init_cond
        else:
            return [1.] + [0.] * 7

    @staticmethod
    def __rmse_loss(a, b):
        return torch.sqrt(
            0.5 * torch.mean(
                torch.pow(a - b, 2)
            )
        )

    @staticmethod
    def __loss_gte_one(parameter):
        return torch.where(parameter.ge(1.), torch.ones(1), torch.zeros(1)) * parameter.abs()

    @staticmethod
    def __loss_lte_zero(parameter: torch.Tensor):
        return torch.where(parameter.le(0.), torch.ones(1), torch.zeros(1)) * parameter.abs()

    def first_derivative_loss(self):
        loss_1st_derivative_total = torch.zeros(1)
        if self.der_1st_reg != 0:
            for key, value in self._params.items():
                first_derivative = derivatives.first_derivative(value, self.sample_time)
                loss_1st_derivative_total = loss_1st_derivative_total + 0.5 * torch.pow(first_derivative, 2)

        return self.der_1st_reg * loss_1st_derivative_total

    def second_derivative_loss(self):
        loss_2nd_derivative_total = torch.zeros(1)
        for key, value in self._params.items():
            first_derivative = derivatives.second_derivative(value, self.sample_time)
            loss_2nd_derivative_total = loss_2nd_derivative_total + 0.5 * torch.pow(first_derivative, 2)

        return loss_2nd_derivative_total

    def bound_parameter_regularization(self):
        bound_reg_total = torch.zeros(1)
        for key, value in self._params.items():
            bound_reg = self.__loss_gte_one(value) + self.__loss_lte_zero(value)
            bound_reg_total = bound_reg_total + bound_reg

        return self.bound_reg * bound_reg_total

    def losses(self, inferences, targets) -> Dict:

        # this function converts target values to torch.tensor with specified dtype
        def to_torch_float(target):
            if isinstance(target, numpy.ndarray) or isinstance(target, list):
                return torch.tensor(target, dtype=self.dtype)
            return target.to(dtype=self.dtype)

        # uses to_torch_float to convert given targets
        targets = {key: to_torch_float(value) for key, value in targets.items()}

        d_rmse_loss = self.__rmse_loss(targets["d"], inferences["d"])
        r_rmse_loss = self.__rmse_loss(targets["r"], inferences["r"])
        t_rmse_loss = self.__rmse_loss(targets["t"], inferences["t"])
        h_rmse_loss = self.__rmse_loss(targets["h_detected"], inferences["h_detected"])
        e_rmse_loss = self.__rmse_loss(targets["e"], inferences["e"])

        total_rmse = self.d_weight * d_rmse_loss + self.r_weight * r_rmse_loss + self.t_weight * t_rmse_loss \
            + self.h_weight * h_rmse_loss + self.e_weight * e_rmse_loss

        der_1st_loss = self.first_derivative_loss()
        der_2nd_loss = self.second_derivative_loss()

        bound_reg = self.bound_parameter_regularization()

        total_loss = total_rmse + der_1st_loss

        return {
            self.val_loss_checked: total_rmse,
            "d_rmse": d_rmse_loss,
            "r_rmse": r_rmse_loss,
            "t_rmse": t_rmse_loss,
            "h_rmse": h_rmse_loss,
            "e_rmse": e_rmse_loss,
            "der_1st": der_1st_loss,
            "der_2nd": der_2nd_loss,
            "bound_reg": bound_reg,
            self.backward_loss_key: total_loss
        }

    def inference(self, time_grid) -> Dict:
        sol = self.integrate(time_grid)
        s = sol[:,0]
        i = sol[:,1]
        d = sol[:,2]
        a = sol[:, 3]
        r = sol[:, 4]
        t = sol[:, 5]
        e = self.init_cond[7] + torch.cumsum(t, dim=0)
        h = self.population - (s + i + d + a + r + t + e)

        h_recovered = self.init_cond[6] + torch.cumsum(
            self.rho * d + self.zeta * r + self.sigma * t,
            dim=0
        )

        return {
            "s": s,
            "i": i,
            "d": d,
            "a": a,
            "r": r,
            "t": t,
            "h": h,
            "e": e,
            "h_recovered": h_recovered,
            "sol": sol
        }

    # TODO
    @classmethod
    def init_trainable_model(cls, initial_params: dict, initial_conditions, integrator, **model_params):
        time_step = model_params["time_step"]
        population = model_params["population"]

        d_weight = model_params.get("d_weight", 0.)
        r_weight = model_params.get("r_weight", 0.)
        t_weight = model_params.get("t_weight", 0.)
        h_weight = model_params.get("h_weight", 0.)
        e_weight = model_params.get("e_weight", 1.)

        der_1st_reg = model_params.get("der_1st_reg", 2e4)

        bound_reg = model_params.get("bound_reg", 1e5)

        return Sidharte(initial_params, population, initial_conditions, integrator, time_step,
                        d_weight=d_weight,
                        r_weight=r_weight,
                        t_weight=t_weight,
                        h_weight=h_weight,
                        e_weight=e_weight,
                        der_1st_reg=der_1st_reg,
                        bound_reg=bound_reg
                        )


    # TODO
    @classmethod
    def init_optimizers(cls, model: AbstractModel, learning_rates: dict, optimizers_params: dict) -> List[Optimizer]:
        lr_alpha = learning_rates["alpha"]

        m = optimizers_params.get("m", 1 / 9)
        a = optimizers_params.get("a", 0.05)
        momentum = optimizers_params.get("momentum", True)

        optimizer = NewSirOptimizer(model.params, learning_rates, m=m, a=a, momentum=momentum)
        return [optimizer]

    @classmethod
    def update_optimizers(cls, optimizers, model, learning_rates: dict):
        pass

    @staticmethod
    def compute_initial_conditions_from_targets(targets: dict, model_params: dict):

        """
        targets = {
            "d": "isolamento_domiciliare",
            "r": "ricoverati_con_sintomi",
            "t": "terapia_intensiva",
            "h_detected": "dimessi_guariti",
            "e": "deceduti"
        }
        """

        population = model_params["population"]

        D0 = targets["d"][0].item()  # isolamento
        R0 = targets["r"][0].item()  # ricoverati con sintomi
        T0 = targets["t"][0].item()  # terapia intensiva
        H0 = targets["h_detected"][0].item()  # dimessi guariti
        E0 = targets["e"][0].item()  # deceduti

        # for now we assume that the number of undetected is equal to the number of detected
        # meaning that half of the infectious were not detected
        I0 = D0  # isolamento domiciliare
        A0 = R0  # ricoverati con sintomi

        # we could consider them to be 0, alternatively
        # I0 = 0.  # isolamento domiciliare
        # A0 = 0.  # ricoverati con sintomi
        # TODO: maybe there are better options?

        S0 = population - (I0 + D0 + A0 + R0 + T0 + H0 + E0)

        return (
            S0,
            I0,
            D0,
            A0,
            R0,
            T0,
            H0,
            E0
        )