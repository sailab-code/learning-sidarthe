import abc


class Integrator(metaclass=abc.ABCMeta):
    """
    Abstract Class for Integrators.
    """
    def __init__(self, time_step=1.):
        """
        Integrator constructor.

        :param time_step:
        """
        self.time_step = time_step

    @abc.abstractmethod
    def __call__(self, diff_equations, initial_conditions, time_grid):
        """

        :param diff_equations:
        :param initial_conditions:
        :param time_grid:

        :return:
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}:{self.time_step}"

    def to_dict(self):
        return {
            "class": self.__class__.__name__,
            "time_step": self.time_step
        }