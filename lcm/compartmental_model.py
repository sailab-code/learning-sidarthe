import abc
import torch
import pytorch_lightning as pl

from typing import List, Dict


class CompartmentalModel(pl.LightningModule, metaclass=abc.ABCMeta):
    """
    Compartmental Model abstract class from which other classes should extend
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.initial_conditions = kwargs["initial_conditions"]
        self.time_step = kwargs["time_step"]
        self.integrator = kwargs["integrator"](self.time_step)

    def integrate(self, time_grid):
        """
        Integrate ODE on the given time interval time_grid.
        :param time_grid: time interval.
        :return: the solution of the ODE in time_grid.
        """
        return self.integrator(self.differential_equations, torch.tensor([self.initial_conditions]), time_grid)

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
        Computation of reproduction number R_t at each
        time step of a given time interval.
        :param time_grid: A torch tensor of shape T
        with the time interval where to compute R_t.
        :return: A tensor with R(t), with t in [0,...,T].
        """
        pass