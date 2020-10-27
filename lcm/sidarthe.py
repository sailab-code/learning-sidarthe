import copy

import torch

from typing import List, Dict, Union
from .compartmental_model import CompartmentalModel


class Sidarthe(CompartmentalModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.EPS = kwargs['EPS']
        self._params = {key: torch.tensor(value, requires_grad=True) for key, value in
                        kwargs["params"].items()}
        self.tied_parameters = kwargs["tied_parameters"]
        self.loss_fn = kwargs["loss_fn"]
        self.regularization_fn = kwargs["regularization"]
        self.population = kwargs["population"]

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
    def param_groups(self) -> Dict:
        return {
            "infection_rates": ('alpha', 'beta', 'gamma', 'delta'),
            "detection_rates": ('epsilon', 'theta'),
            "symptoms_development_rates": ('eta', 'zeta'),
            "acute_symptoms_development_rates": ('mu', 'nu'),
            "recovery_rates": ('kappa', 'lambda', 'xi', 'rho', 'sigma'),
            "death_rates": tuple(['tau'])
        }

    def extend_param(self, value, length):
        len_diff = length - value.shape[0]
        ext_tensor = torch.tensor([value[-1] for _ in range(len_diff)], dtype=self.dtype)
        ext_tensor = torch.cat((value, ext_tensor))
        rectified_param = torch.relu(ext_tensor)
        return torch.where(rectified_param >= self.EPS, rectified_param, rectified_param + self.EPS)

    def rectify_param(self, param):
        rectified = torch.relu(param)
        return torch.where(rectified >= self.EPS, rectified, self.EPS)

    @staticmethod
    def get_param_at_t(param, _t):
        _t = _t.long()
        if 0 <= _t < param.shape[0]:
            return param[_t].unsqueeze()
        else:
            return param[_t].unsqueeze().detach()  # TODO: check this detach()

    def __getattr__(self, item):
        """
        Used to quickly obtain a single parameter
        :param item: name of the parameter
        :return: parameter tensor
        """
        if item.replace("_", "") in self._params.keys():
            if item in self.tied_parameters.keys():
                param = self._params[self.tied_parameters[item]]
            else:
                param = self._params[item]

            return self.rectify_param(param)
        else:
            raise AttributeError

    def forward(self, time_grid):
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

        extended_params = {key: self.extend_param(value, time_grid.shape[0]) for key, value in self.params.items()}

        # region compute R0
        c1 = extended_params['epsilon'] + extended_params['zeta'] + extended_params['lambda']
        c2 = extended_params['eta'] + extended_params['rho']
        c3 = extended_params['theta'] + extended_params['mu'] + extended_params['kappa']
        c4 = extended_params['nu'] + extended_params['xi']

        r0 = extended_params['alpha'] + extended_params['beta'] * extended_params['epsilon'] / c2
        r0 = r0 + extended_params['gamma'] * extended_params['zeta'] / c3
        r0 = r0 + extended_params['delta'] * (extended_params['eta'] * extended_params['epsilon']) / (c2 * c4)
        r0 = r0 + extended_params['delta'] * extended_params['zeta'] * extended_params['theta'] / (c3 * c4)
        r0 = r0 / c1
        # endregion

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
            "r0": r0
        }

    def differential_equations(self, t, x):
        p = {key: self.get_param_at_t(param, t) for key, param in self.params.items()}

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

    def training_step(self, batch, batch_idx):
        t_grid, targets = batch
        hats = self.forward(t_grid)
        target_losses = self.loss_fn(hats, targets)
        regularization_loss = self.regularization_fn(self.params)

        # TODO: add log code

        return target_losses["backward"] + regularization_loss["backward"]
