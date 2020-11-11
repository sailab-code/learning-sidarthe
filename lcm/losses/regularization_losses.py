import torch

from lcm.losses import RegularizationLoss


class LteZero(RegularizationLoss):
    """
    Regularization Term to bound parameters in the positive set (0, inf).
    """
    def param_loss(self, param):
        return param.abs() * torch.where(param.le(0.), torch.ones(1), torch.zeros(1))


class LogVicinity(RegularizationLoss):
    """
    Log-based Regularization loss to bound parameters in the positive set (0,inf).
    The closer to 0, the higher the penalty.
    """
    def __init__(self, weight, denominator=0.5e3, exponent=10, **kwargs):
        super().__init__(weight, **kwargs)
        self.denominator = denominator
        self.exponent = exponent

    def param_loss(self, param):
        return torch.pow(
            torch.log(param) / torch.log(torch.tensor(1. / self.denominator)),
            self.exponent
        )


class FirstDerivative(RegularizationLoss):
    """
    Regularization loss to enforce smooth parameter functions.
    In the reference paper, it is indicated as |u_dot(t)|^2.
    Compute the discrete first derivative of a parameter wrt to time t.
    """
    def __init__(self, weight, time_step, **kwargs):
        super().__init__(weight, **kwargs)
        self.time_step = time_step

    @staticmethod
    def first_derivative_central(f_x_plus_h, f_x_minus_h, h):
        return (f_x_plus_h - f_x_minus_h) / (2 * h)

    @staticmethod
    def first_derivative_forward(f_x_plus_h, f_x, h):
        return (f_x_plus_h - f_x) / h

    @staticmethod
    def first_derivative_backward(f_x, f_x_minus_h, h):
        return (f_x - f_x_minus_h) / h

    def param_loss(self, param):
        if param.shape[0] < 3:  # must have at least 3 values to properly compute first derivative
            return torch.tensor(0.)

        forward = self.first_derivative_forward(param[1], param[0], self.time_step).unsqueeze(0)
        central = self.first_derivative_central(param[2:], param[:-2], self.time_step)
        backward = self.first_derivative_backward(param[-1], param[-2], self.time_step).unsqueeze(0)

        return torch.cat((forward, central, backward), dim=0)
