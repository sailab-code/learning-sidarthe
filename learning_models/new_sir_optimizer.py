from typing import List, Dict

import torch
from torch.optim.optimizer import Optimizer

from utils.visualization_utils import Curve, generic_plot


class NewSirOptimizer(Optimizer):
    def __init__(self, params: Dict, learning_rates: Dict, momentum=True, m=None, a=None, summary=None):

        if momentum is True and (m is None or a is None):
            raise ValueError("Must specify m and a if momentum is True")

        self.momentum = momentum
        self.m = m
        self.a = a
        self.summary = summary
        self.epoch = 1

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
            param_name = group["name"]
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
#                        momentum_term = -eta[t] * d_p[t] + 0.2 * update[t - 1]
                        momentum_term = -eta[t] * d_p[t] + mu[t] * update[t - 1]
                        update.append(momentum_term)
                    update = torch.tensor(update)
                else:
                    update = -lr * d_p

                if self.summary is not None and self.epoch % 50 == 0:
                    pl_x = range(0, update.shape[0])
                    grad_curve = Curve(pl_x, update.detach().numpy(), '.', label=f"{param_name}", color=None)
                    fig = generic_plot([grad_curve], f"{param_name} update over time", None)
                    self.summary.add_figure(f"grads/{param_name}", fig, global_step=self.epoch)
                parameter.data.add_(update)

        self.epoch += 1
