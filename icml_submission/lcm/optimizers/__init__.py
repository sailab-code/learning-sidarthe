from typing import Dict

import torch
from torch.optim.optimizer import Optimizer


class MomentumOptimizer(Optimizer):
    """
    Optimizer implementing temporal momentum, defined in:
        Zugarini, Andrea, Enrico Meloni, et al.
        "An Optimal Control Approach to Learning in SIDARTHE Epidemic model."
        arXiv preprint arXiv:2010.14878 (2020).
    """

    def __init__(self, params: Dict, learning_rates: Dict, momentum_settings: Dict = {}):
        """
        Initialize the optimizer with the learning parameters (params),
        their learning rates (learning_rates) and
        the momentum hyperparameters (momentum_settings).
        :param params: A Dict of parameters
        :param learning_rates:
        :param momentum_settings: A Dict with hyperparameters of momentum. Its values are:
            'a' (defaults to 0.0)
            'b' (defaults to 0.1)
        """

        momentum = momentum_settings.get('active', True)
        a = momentum_settings.get('a', 0.)
        b = momentum_settings.get('b', 0.1)

        if momentum is True and (b is None or a is None):
            raise ValueError("Must specify b and a if momentum is True")

        params_list = []
        for key, value in params.items():
            param_dict = {
                "params": value,
                "name": key,
                "lr": learning_rates[key]
            }
            params_list.append(param_dict)

        defaults = {
            "momentum": momentum,
            "a": a,
            "b": b
        }
        super(MomentumOptimizer, self).__init__(params_list, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            a = group["a"]
            b = group["b"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                d_p = parameter.grad
                if momentum:
                    times = torch.arange(parameter.shape[0], dtype=parameter.dtype)
                    mu = torch.sigmoid(b * times)
                    eta = lr / (1 + a * times)
                    update = [-eta[0] * d_p[0]]
                    for t in range(1, d_p.size(0)):
                        momentum_term = -eta[t] * d_p[t] + mu[t] * update[t - 1]
                        update.append(momentum_term)
                    # update = torch.tensor(update, device=parameter.device, dtype=parameter.dtype)

                    # update can be a list of multi-valued tensors
                    update = torch.cat(update, dim=0).reshape(d_p.size(0), -1)

                else:
                    update = -lr * d_p
                parameter.data.add_(update)

        return loss