import abc
import torch
import pytorch_lightning as pl

from typing import List, Dict, Union, TextIO

from dataset.spatio_temporal_dataset import SpatioTemporalSidartheDataset
from .optimizers import MomentumOptimizer
import json

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

        self.EPS = kwargs.get('EPS', 1e-6)

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
        :param x: list, value of the state variables after previous step.

        :return: result of ODE at time t.
        """
        pass

    @abc.abstractmethod
    def get_rt(self, time_grid):
        """
        Computation of basic reproduction number R_t at each
        time step of a given time interval.

        :param time_grid: A torch tensor of shape T with the time interval where to compute R_t.

        :return: A tensor with R(t), with t in [0,...,T].
        """
        pass

    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        :return:
        """
        t_grid, targets, train_mask = batch
        t_grid = t_grid.squeeze(0)
        targets = {key: target.squeeze(0) for key, target in targets.items()}
        hats = self.forward(t_grid)
        target_losses = self.loss_fn(hats, targets, train_mask)
        regularization_loss = self.regularization_fn(self.params)

        for k,v in target_losses.items():
            self.log(f"train_loss_{k}", v, prog_bar=("weighted" in k))

        return target_losses["weighted"] + regularization_loss["weighted"]

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        :return:
        """
        t_grid, targets, validation_mask = batch
        t_grid = t_grid.squeeze(0)
        targets = {key: target.squeeze(0) for key, target in targets.items()}
        hats = self.forward(t_grid)
        target_losses = self.loss_fn(hats, targets, validation_mask)
        regularization_loss = self.regularization_fn(self.params)

        for k, v in target_losses.items():
            self.log(f"val_loss_{k}", v, prog_bar=True)

        return {
            "hats": hats,
            "targets": targets,
            "target_loss": target_losses['unweighted'],
            "regularization_loss": regularization_loss['unweighted']
        }

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        :return:
        """
        metrics = self.validation_step(batch, batch_idx)
        return metrics

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization. We set the :class:`~lcm.optimizers.MomentumOptimizer` as default.
        Override this to change it.

        :return: The :class:`~lcm.optimizers.MomentumOptimizer` instance.
        """

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

    def get_description(self):
        return {
            "class": self.__class__.__name__,
            "initial_conditions": self.initial_conditions.tolist(),
            "integrator": self.integrator.to_dict(),
            "time_step": self.time_step,
            "EPS": self.EPS,
            "params": {k: v.detach().tolist() for k, v in self.params.items()}
        }

    def __str__(self):
        return self.__class__.__name__



