from functools import reduce
from typing import List

import numpy
import torch
from torch.optim.optimizer import Optimizer

from learning_models.abstract_model import AbstractModel
import utils.derivatives as derivatives
from learning_models.sir_optimizer import SirOptimizer


class NewSir(AbstractModel):
    def __init__(self, beta, gamma, delta, population, init_cond, integrator, sample_time, **kwargs):
        super().__init__(init_cond, integrator, sample_time)
        self.population = population
        self._params = {
            "beta": torch.tensor(beta, requires_grad=True),
            "gamma": torch.tensor(gamma, requires_grad=True),
            "delta": torch.tensor(delta, requires_grad=True)
        }

        self.b_reg = kwargs.get("b_reg", 1e5)
        self.c_reg = kwargs.get("c_reg", 1e5)
        self.d_reg = kwargs.get("d_reg", 1e9)

        self.der_1st_reg = kwargs.get("der_1st_reg", 0.)
        self.der_2nd_reg = kwargs.get("der_2nd_reg", 0.)

        self.y_loss_weight = kwargs.get("y_loss_weight", 0.0)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    def differential_equations(self, t, x):
        X_t = x
        t = t.long()
        beta = self.params["beta"]
        gamma = self.params["gamma"]

        if 0 < t < beta.shape[0]:
            beta = beta[t] / self.population
        else:
            beta = beta[-1] / self.population

        if 0 < t < gamma.shape[0]:
            gamma = gamma[t]
        else:
            gamma = gamma[-1]

        beta = beta.unsqueeze(0)
        gamma = gamma.unsqueeze(0)

        return torch.cat((
            - beta * X_t[0] * X_t[1],
            beta * X_t[0] * X_t[1] - gamma * X_t[1],
            gamma * X_t[1]
        ), dim=0)

    def omega(self, t):
        if t >= 0:
            return self.init_cond
        else:
            return [1, 0, 0]

    def first_derivative_loss(self):
        if self.der_1st_reg != 0:

            loss_1st_derivative_beta = derivatives.first_derivative(self.params["beta"], self.sample_time)
            loss_1st_derivative_gamma = derivatives.first_derivative(self.params["gamma"], self.sample_time)
            loss_1st_derivative_delta = derivatives.first_derivative(self.params["delta"], self.sample_time)
            loss_1st_derivative_total = (
                    loss_1st_derivative_beta + loss_1st_derivative_gamma + loss_1st_derivative_delta
            )

            return self.der_1st_reg * torch.mean(loss_1st_derivative_total)
        else:
            return torch.zeros(1)

    def second_derivative_loss(self):
        if self.der_1st_reg != 0:
            loss_2nd_derivative_beta = derivatives.second_derivative(self.params["beta"], self.sample_time)
            loss_2nd_derivative_gamma = derivatives.second_derivative(self.params["gamma"], self.sample_time)
            loss_2nd_derivative_delta = derivatives.second_derivative(self.params["delta"], self.sample_time)
            loss_2nd_derivative_total = (
                    loss_2nd_derivative_beta + loss_2nd_derivative_gamma + loss_2nd_derivative_delta
            )

            return self.der_2nd_reg * torch.mean(loss_2nd_derivative_total)
        else:
            return torch.zeros(1)

    @staticmethod
    def __loss_gte_one(parameter):
        return torch.where(parameter.ge(1.), torch.ones(1), torch.zeros(1)) * parameter.abs()

    @staticmethod
    def __loss_lte_zero(parameter: torch.Tensor):
        return torch.where(parameter.le(0.), torch.ones(1), torch.zeros(1)) * parameter.abs()

    def losses(self, inferences, targets):
        def mapper(target):
            if isinstance(target, numpy.ndarray) or isinstance(target, list):
                return torch.tensor(target)

        def mse_loss(a, b):
            return torch.sqrt(
                0.5 * torch.mean(
                    torch.pow(a - b, 2)
                )
            )

        targets = {key: mapper(value) for key, value in targets}

        w_target = targets["w"]
        w_hat = inferences["w"]
        w_mse_loss = mse_loss(w_hat, w_target)

        y_target = targets["y"]
        y_hat = inferences["y"]

        y_mse_loss = 0.
        if y_target is not None:
            y_mse_loss = mse_loss(w_hat, w_target)

        loss_reg_beta = self.b_reg * (
                self.__loss_gte_one(self.params["beta"]) + self.__loss_lte_zero(self.params["beta"]))
        loss_reg_gamma = self.b_reg * (
                self.__loss_gte_one(self.params["gamma"]) + self.__loss_lte_zero(self.params["gamma"]))
        loss_reg_delta = self.b_reg * (
                self.__loss_gte_one(self.params["delta"]) + self.__loss_lte_zero(self.params["delta"]))

        mse = ((1.0 - self.y_loss_weight) * w_mse_loss) + self.y_loss_weight * y_mse_loss
        total_loss = mse_loss + \
                     torch.mean(loss_reg_beta + loss_reg_gamma + loss_reg_delta)

        return {
            "mse": mse,
            "w_mse": w_mse_loss,
            "y_mse": y_mse_loss,
            [self.backward_loss_key]: total_loss
        }

    def inference(self, time_grid):
        sol = self.integrate(time_grid)
        z_hat = sol[:, 2]

        delta = self.params["delta"]
        len_diff = z_hat.shape[0] - delta.shape[0]
        if len_diff > 0:
            delta = torch.cat((delta, delta[-1].expand(len_diff)))

        w_hat = delta * z_hat

        return {
            "x": sol[:, 0],
            "y": sol[:, 1],
            "z": sol[:, 2],
            "w": w_hat,
            "sol": sol
        }

    @classmethod
    def init_trainable_model(cls, initial_params: dict, initial_conditions, integrator, **model_params):

        beta = initial_params["beta"]
        gamma = initial_params["gamma"]
        delta = initial_params["delta"]
        population = model_params["population"]
        t_inc = model_params["t_inc"]

        b_reg = model_params.get("b_reg", 1e7)
        c_reg = model_params.get("c_reg", 1e7)
        d_reg = model_params.get("d_reg", 1e8)

        der_1st_reg = model_params.get("der_1st_reg", 1e3)
        der_2nd_reg = model_params.get("der_2nd_reg", 1e3)

        y_loss_weight = model_params.get("y_loss_weight", 0.0)

        return NewSir(beta, gamma, delta, population, initial_conditions,
                      b_reg=b_reg, c_reg=c_reg, d_reg=d_reg,
                      der_1st_reg=der_1st_reg, der_2nd_reg=der_2nd_reg,
                      sample_time=t_inc, y_loss_weight=y_loss_weight,
                      integrator=integrator)

    @classmethod
    def init_optimizers(cls, model: AbstractModel, optimizer_params: dict, learning_rates: dict) -> List[Optimizer]:
        lr_b = learning_rates["beta"]
        lr_g = learning_rates["gamma"]
        lr_d = learning_rates["delta"]

        m = optimizer_params.get("m", 1 / 9)
        a = optimizer_params.get("a", 3.0)
        b = optimizer_params.get("b", 0.05)
        momentum = optimizer_params.get("momentum", True)

        optimizer = SirOptimizer(model.trainable_parameters, [lr_b, lr_g, lr_d],
                                 m=m, a=a, b=b,
                                 momentum=momentum)
        return [optimizer]

    @classmethod
    def update_optimizers(cls, optimizers, model, learning_rates: dict):
        optimizer = optimizers[0]
        lr_b = learning_rates["beta"]
        lr_g = learning_rates["gamma"]
        lr_d = learning_rates["delta"]
        optimizer.etas = [lr_b, lr_g, lr_d]

    @staticmethod
    def compute_initial_conditions_from_targets(targets: dict, model_params: dict):
        population = model_params["population"]
        epsilon = targets["y"][0].item() / population
        epsilon_z = targets["w"][0].item() / population
        S0 = 1 - (epsilon + epsilon_z)
        I0 = epsilon
        S0 = S0 * population
        I0 = I0 * population
        Z0 = epsilon_z

        return S0, I0, Z0
