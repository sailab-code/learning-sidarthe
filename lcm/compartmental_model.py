import abc
import torch
import pytorch_lightning as pl

from typing import List, Dict
from .optimizers import MomentumOptimizer


class CompartmentalModel(pl.LightningModule, metaclass=abc.ABCMeta):
    """
    Compartmental Model abstract class from which other classes should extend
    """

    def __init__(self, **kwargs):
        super().__init__()
        # self.initial_conditions = torch.tensor([kwargs["initial_conditions"]], device=self.device, dtype=self.dtype)
        self.initial_conditions = kwargs["initial_conditions"]
        self.time_step = kwargs["time_step"]
        self.integrator = kwargs["integrator"](self.time_step)

    def integrate(self, time_grid):
        """
        Integrate ODE on the given time interval time_grid.
        :param time_grid: time interval.
        :return: the solution of the ODE in time_grid.
        """
        return self.integrator(self.differential_equations,
                               self.initial_conditions,
                               time_grid)

    @property
    @abc.abstractmethod
    def params(self) -> Dict:
        pass

    @params.setter
    @abc.abstractmethod
    def params(self, value: Dict):
        pass

    @abc.abstractmethod
    def differential_equations(self, t, x):
        """
        Definition of ODE.
        :param t: int, the time step.
        :param x: list, value of the state variables
        after previous step.
        :return: result of ODE at time t.
        """
        pass

    @abc.abstractmethod
    def get_rt(self, time_grid):
        """
        Computation of basic reproduction number R_t at each
        time step of a given time interval.
        :param time_grid: A torch tensor of shape T
        with the time interval where to compute R_t.
        :return: A tensor with R(t), with t in [0,...,T].
        """
        pass

    def training_step(self, batch, batch_idx):
        t_grid, targets, train_mask = batch
        t_grid = t_grid.squeeze(0)
        targets = {key: target.squeeze(0) for key, target in targets.items()}
        hats = self.forward(t_grid)
        target_losses = self.loss_fn(hats, targets, train_mask)
        regularization_loss = self.regularization_fn(self.params)

        for k,v in target_losses.items():
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

    def extend_param(self, value, length):
        len_diff = length - value.shape[0]
        ext_tensor = torch.tensor([value[-1] for _ in range(len_diff)], device=self.device, dtype=self.dtype)
        ext_tensor = torch.cat((value, ext_tensor))
        return ext_tensor[:length]

    def rectify_param(self, param):
        rectified = torch.relu(param)
        # noinspection PyTypeChecker
        return torch.where(rectified >= self.EPS,
                           rectified,
                           torch.full_like(rectified, self.EPS,
                                           device=self.device,
                                           dtype=self.dtype
                                           )
                           )

    def __str__(self):
        return self.__class__.__name__
