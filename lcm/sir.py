import copy

import torch

from typing import List, Dict, Union, Optional, Sequence, Tuple

from .optimizers import MomentumOptimizer

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

    def extend_param(self, value, length):
        len_diff = length - value.shape[0]
        ext_tensor = torch.tensor([value[-1] for _ in range(len_diff)], dtype=self.dtype)
        ext_tensor = torch.cat((value, ext_tensor))
        return ext_tensor[:length]
        # rectified_param = torch.relu(ext_tensor)
        # return torch.where(rectified_param >= self.EPS, rectified_param, rectified_param + self.EPS)

    def rectify_param(self, param):
        rectified = torch.relu(param)
        return torch.where(rectified >= self.EPS, rectified, torch.full_like(rectified, self.EPS))

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
        extended_params = {key: self.extend_param(value, time_grid.shape[0]) for key, value in self.params.items()}

        return extended_params["beta"] / extended_params["gamma"]

    def differential_equations(self, t, x):
        p = {key: self.get_param_at_t(key, t) for key in self.params.keys()}

        S = x[0]
        I = x[1]

        S_dot = - p['beta']*(S * I)/self.population
        I_dot = -S_dot - (p['gamma'] * I)
        R_dot = p['gamma'] * I

        return torch.cat((
            S_dot,
            I_dot,
            R_dot,
        ), dim=0)

    def training_step(self, batch, batch_idx):
        t_grid, targets, train_mask = batch
        t_grid = t_grid.squeeze(0)
        targets = {key: target.squeeze(0) for key, target in targets.items()}
        hats = self.forward(t_grid)
        target_losses = self.loss_fn(hats, targets, train_mask)
        regularization_loss = self.regularization_fn(self.params)

        for k, v in target_losses.items():
            self.log(k, v, prog_bar=True)

        return target_losses["backward"] + regularization_loss["backward"]

    def validation_step(self, batch, batch_idx):
        t_grid, targets, validation_mask = batch
        t_grid = t_grid.squeeze(0)
        targets = {key: target.squeeze(0) for key, target in targets.items()}
        hats = self.forward(t_grid)
        target_losses = self.loss_fn(hats, targets, validation_mask)
        regularization_loss = self.regularization_fn(self.params)

        for k, v in target_losses.items():
            self.log(k, v, prog_bar=True)

        return {
            "hats": hats,
            "targets": targets,
            "target_loss": target_losses['validation'],
            "regularization_loss": regularization_loss['validation']
        }

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        return metrics

    def configure_optimizers(self):
        return MomentumOptimizer(self.trainable_params, self.learning_rates, self.momentum_settings)
