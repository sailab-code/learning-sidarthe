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

        self.momentum = momentum_settings.get('active', True)
        self.a = momentum_settings.get('a', 0.)
        self.b = momentum_settings.get('b', 0.1)

        if self.momentum is True and (self.b is None or self.a is None):
            raise ValueError("Must specify m and a if momentum is True")

        params_list = []
        for key, value in params.items():
            param_dict = {
                "params": value,
                "name": key,
                "lr": learning_rates[key]
            }
            params_list.append(param_dict)

        defaults = dict()
        super().__init__(params_list, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            for idx, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                d_p = parameter.grad
                if self.momentum:
                    times = torch.arange(parameter.shape[0], dtype=parameter.dtype)
                    mu = torch.sigmoid(self.b * times)
                    eta = lr / (1 + self.a * times)
                    update = [-eta[0] * d_p[0]]
                    for t in range(1, d_p.size(0)):
                        momentum_term = -eta[t] * d_p[t] + mu[t] * update[t - 1]
                        update.append(momentum_term)
                    update = torch.tensor(update)
                else:
                    update = -lr * d_p

                parameter.data.add_(update)
