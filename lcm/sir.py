import copy

import torch

from typing import Dict

from .compartmental_model import CompartmentalModel


class SIR(CompartmentalModel):
    """
    SIR Compartmental Model
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.EPS = kwargs.get('EPS', 0.)
        self._params = {key: torch.tensor(value, requires_grad=True) for key, value in
                        kwargs["params"].items()}
        self.loss_fn = kwargs["loss_fn"]
        self.regularization_fn = kwargs["regularization"]
        self.population = kwargs["population"]
        self.learning_rates = kwargs.get("learning_rates", {})
        self.momentum_settings = kwargs.get("momentum_settings", {})

    @property
    def params(self) -> Dict:
        return {
            "beta": self.beta,
            "gamma": self.gamma
        }

    @params.setter
    def params(self, value: Dict):
        self._params = copy.deepcopy(value)

    @property
    def trainable_params(self) -> Dict:
        return {
            key: param
            for key, param in self._params.items()
        }

    @property
    def param_groups(self) -> Dict:
        return {
            "infection_rates": ('beta',),
            "recovery_rates": ('gamma',),
        }

    def get_param_at_t(self, param_key, _t):
        param = self.params[param_key]
        _t = _t.long()
        if 0 <= _t < param.shape[0]:
            p = param[_t].unsqueeze(0)
        else:
            p = param[-1].unsqueeze(0).detach()  # TODO: check this detach()

        return p

    def __getattr__(self, item):
        """
        Used to quickly obtain a single parameter

        :param item: name of the parameter

        :return: parameter tensor
        """

        param_key = item.replace("_", "")

        if param_key in self._params.keys():
            param = self._params[param_key]
            return self.rectify_param(param)
        else:
            raise AttributeError

    def forward(self, time_grid):
        """
        Implements the predictions of SIR for the input time_grid.

        :param time_grid: (Tensor) the input time grid.

        :return: A dictionary containing all the states of SIR and the r0 calculated during all the time period.
        """

        sol = self.integrate(time_grid)
        s = sol[:, 0]
        i = sol[:, 1]
        r = self.population - (s + i)

        rt = self.get_rt(time_grid)

        return {
            "s": s,
            "i": i,
            "r": r,
            "r0": rt
        }

    def get_rt(self, time_grid):
        r"""
        Computation of basic reproduction number R_t at each
        time step of a given time interval.

        :param time_grid: A torch tensor of shape T with the time interval where to compute R_t.

        :return: A tensor with R(t), with t in [0,...,T].

        Code:

            .. code-block:: python

                r_t = extended_params['beta'] / extended_params['gamma']
        """

        extended_params = {key: self.extend_param(value, time_grid.shape[0]) for key, value in self.params.items()}

        return extended_params["beta"] / extended_params["gamma"]

    def differential_equations(self, t, x):
        p = {key: self.get_param_at_t(key, t) for key in self.params.keys()}

        S = x[0]
        I = x[1]

        S_dot = - p['beta'] * (S * I) / self.population
        I_dot = -S_dot - (p['gamma'] * I)
        R_dot = p['gamma'] * I

        return torch.cat((
            S_dot,
            I_dot,
            R_dot,
        ), dim=0)
