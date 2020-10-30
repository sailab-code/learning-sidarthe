import abc


class Integrator(metaclass=abc.ABCMeta):
    def __init__(self, time_step=1.):
        self.time_step = time_step

    @abc.abstractmethod
    def __call__(self, diff_equations, initial_conditions, time_grid):
        pass