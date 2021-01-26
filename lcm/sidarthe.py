import copy

from torch import optim
import torch
from torch.nn import Parameter

from typing import Dict

from .compartmental_model import CompartmentalModel

DEFAULT_LR = 1e-5
DEFAULT_INITIAL_PARAMS = {
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

class Sidarthe(CompartmentalModel):
    """
    SIDARTHE Compartmental Model:
        Giordano, Giulia, et al.
        "Modelling the COVID-19 epidemic and implementation of population-wide interventions in Italy."
        Nature Medicine (2020): 1-6.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.EPS = kwargs.get('EPS', 1e-6)

        self.tied_parameters = kwargs.get("tied_parameters", {})

        self._params = {}
        initial_params = kwargs.get("params", self.get_default_initial_params())
        for key, param_val in initial_params.items():
            if key not in self.tied_parameters.keys():
                param = Parameter(torch.tensor(param_val, device=self.device))
                self.register_parameter(key, param)
                self._params[key] = param

        self.loss_fn = kwargs["loss_fn"]
        self.regularization_fn = kwargs["reg_fn"]
        self.population = torch.tensor(kwargs["population"], requires_grad=False)
        self.learning_rates = kwargs.get("learning_rates", self.get_default_learning_rates())
        self.momentum_settings = kwargs.get("momentum_settings", {})

    def to(self, device):
        """
        Moves and/or casts all model parameters to the device in-place.

        :param device: the device

        :return: self
        """

        super().to(device)
        self.population.to(device)

    def cuda(self, device):
        """
        Moves all model parameters and buffers to the GPU.
        :param device: if specified, all parameters will be copied to that device
        :return: self
        """

        super().cuda(device)
        self.population = self.population.cuda(device)
        self.initial_conditions = self.initial_conditions.cuda(device)

    def cpu(self):
        """
        Moves parameters and buffers to CPU
        """

        super().cpu()
        self.population = self.population.cpu()
        self.initial_conditions = self.initial_conditions.cpu()


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
        }

    @params.setter
    def params(self, value: Dict):
        self._params = copy.deepcopy(value)

    @property
    def trainable_params(self) -> Dict:
        return {
            key: param
            for key, param in self._params.items()
            if key not in self.tied_parameters.keys()
        }

    @property
    def param_groups(self) -> Dict:
        return {
            "infection_rates": ('alpha', 'beta', 'gamma', 'delta'),
            "detection_rates": ('epsilon', 'theta'),
            "symptoms_development_rates": ('eta', 'zeta'),
            "acute_symptoms_development_rates": ('mu', 'nu'),
            "recovery_rates": ('kappa', 'lambda', 'xi', 'rho', 'sigma'),
            "death_rates": tuple(['tau'])
        }

    def get_param_at_t(self, param_key, _t):
        param = self.params[param_key]
        _t = _t.long()
        if 0 <= _t < param.shape[0]:
            p = param[_t].unsqueeze(0)
        else:
            p = param[-1].unsqueeze(0).detach()  # TODO: check this detach()

        if param_key in ['alpha', 'beta', 'gamma', 'delta']:  # these params must be divided by population
            return p / self.population
        else:
            return p

    def __getattr__(self, item):
        """
        Used to quickly obtain a single parameter

        :param item: name of the parameter

        :return: parameter tensor
        """

        if item == "params":
            raise RuntimeError("ERROR: Error in calling self.params")

        param_key = item.replace("lambda_", "lambda")

        if param_key in self._params.keys() or param_key in self.tied_parameters.keys():
            if param_key in self.tied_parameters.keys():
                param = self._params[self.tied_parameters[param_key]]
            else:
                param = self._params[param_key]

            return self.rectify_param(param)
        else:
            super().__getattr__(item)

    def forward(self, time_grid):
        """
        Implements the predictions of SIDARTHE model for the input time_grid.

        :param time_grid: (Tensor) the input time grid.

        :return: A dictionary containing all the states of SIDARTHE and the r0 calculated during all the time period.
        """

        sol = self.integrate(time_grid)
        s = sol[:, 0]
        i = sol[:, 1]
        d = sol[:, 2]
        a = sol[:, 3]
        r = sol[:, 4]
        t = sol[:, 5]
        e = sol[:, 6]
        h_detected = sol[:, 7]
        h = self.population - (s + i + d + a + r + t + e)

        rt = self.get_rt(time_grid)

        return {
            "s": s,
            "i": i,
            "d": d,
            "a": a,
            "r": r,
            "t": t,
            "h": h,
            "e": e,
            "h_detected": h_detected,
            "r0": rt
        }

    def get_rt(self, time_grid):
        """
        Computation of basic reproduction number R_t at each
        time step of a given time interval.

        :param time_grid: A torch tensor of shape T with the time interval where to compute R_t.

        :return: A tensor with R(t), with t in [0,...,T].
        """
        extended_params = {key: self.extend_param(value, time_grid.shape[0]) for key, value in self.params.items()}

        # compute R_t
        c1 = extended_params['epsilon'] + extended_params['zeta'] + extended_params['lambda']
        c2 = extended_params['eta'] + extended_params['rho']
        c3 = extended_params['theta'] + extended_params['mu'] + extended_params['kappa']
        c4 = extended_params['nu'] + extended_params['xi']

        rt = extended_params['alpha'] + extended_params['beta'] * extended_params['epsilon'] / c2
        rt = rt + extended_params['gamma'] * extended_params['zeta'] / c3
        rt = rt + extended_params['delta'] * (extended_params['eta'] * extended_params['epsilon']) / (c2 * c4)
        rt = rt + extended_params['delta'] * extended_params['zeta'] * extended_params['theta'] / (c3 * c4)

        return rt / c1

    def differential_equations(self, t, x):
        p = {key: self.get_param_at_t(key, t) for key in self.params.keys()}

        S = x[0]
        I = x[1]
        D = x[2]
        A = x[3]
        R = x[4]
        T = x[5]

        S_dot = -S * (p['alpha'] * I + p['beta'] * D + p['gamma'] * A + p['delta'] * R)
        I_dot = -S_dot - (p['epsilon'] + p['zeta'] + p['lambda']) * I
        D_dot = p['epsilon'] * I - (p['eta'] + p['rho']) * D
        A_dot = p['zeta'] * I - (p['theta'] + p['mu'] + p['kappa']) * A
        R_dot = p['eta'] * D + p['theta'] * A - (p['nu'] + p['xi']) * R
        T_dot = p['mu'] * A + p['nu'] * R - (p['sigma'] + p['tau']) * T
        E_dot = p['tau'] * T
        H_det_dot = p['rho'] * D + p['xi'] * R + p['sigma'] * T

        return torch.cat((
            S_dot,
            I_dot,
            D_dot,
            A_dot,
            R_dot,
            T_dot,
            E_dot,
            H_det_dot
        ), dim=0)

    def get_default_learning_rates(self):
        """
        Setting all the params to the same learning rate value

        :return:
        """
        return {k: DEFAULT_LR for k,v in self._params.items()}

    @staticmethod
    def get_default_initial_params():
        return DEFAULT_INITIAL_PARAMS