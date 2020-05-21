from typing import List, Dict

import torch
from torch.optim.optimizer import Optimizer


class NewSirOptimizer(Optimizer):
    def __init__(self, params: Dict, learning_rates: Dict, momentum=True, m=None, a=None):

        if momentum is True and (m is None or a is None):
            raise ValueError("Must specify m and a if momentum is True")

        self.momentum = momentum
        self.m = m
        self.a = a

        params_list = []
        for key, value in params.items():
            param_dict = {
                "params": value,
                "lr": learning_rates[key]
            }
            params_list.append(param_dict)

        defaults = dict()
        super().__init__(self, params_list, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            for idx, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                d_p = parameter.grad
                if self.momentum:
                    times = torch.arange(parameter.shape[0], dtype=parameter.dtype)
                    mu = torch.sigmoid(self.m * times)
                    eta = lr / (1 + self.a * times)
                    update = [-eta[0] * d_p[0]]
                    for t in range(1, d_p.size(0)):
                        momentum_term = -eta[t] * d_p[t] + mu[t] * update[t - 1]
                        update.append(momentum_term)

                    parameter.add_(torch.tensor(update))
                else:
                    parameter.add_(-lr * d_p)

