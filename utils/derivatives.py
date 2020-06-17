import torch


def second_derivative_central(f_x_plus_h, f_x, f_x_minus_h, h):
    return (f_x_plus_h - 2 * f_x + f_x_minus_h) / (h ** 2)


def second_derivative_forward(f_x_plus_2h, f_x_plus_h, f_x, h):
    return (f_x_plus_2h - 2 * f_x_plus_h + f_x) / (h ** 2)


def second_derivative_backward(f_x, f_x_minus_h, f_x_minus_2h, h):
    return (f_x - 2 * f_x_minus_h + f_x_minus_2h) / (h ** 2)


def second_derivative(parameter: torch.Tensor, sample_time):
    if parameter.shape[0] < 3: # must have at least 3 values to properly compute first derivative
        return torch.tensor(0., dtype=parameter.dtype)

    forward = second_derivative_forward(parameter[2], parameter[1], parameter[0], sample_time).unsqueeze(0)
    central = second_derivative_central(parameter[2:], parameter[1:-1], parameter[:-2], sample_time)
    backward = second_derivative_backward(parameter[-1], parameter[-2], parameter[-3],
                                                 sample_time).unsqueeze(0)
    return torch.cat((forward, central, backward), dim=0)


def first_derivative_central(f_x_plus_h, f_x_minus_h, h):
    return (f_x_plus_h - f_x_minus_h) / (2 * h)


def first_derivative_forward(f_x_plus_h, f_x, h):
    return (f_x_plus_h - f_x) / h


def first_derivative_backward(f_x, f_x_minus_h, h):
    return (f_x - f_x_minus_h) / h


def first_derivative(parameter, sample_time):
    if parameter.shape[0] < 3: # must have at least 3 values to properly compute first derivative
        return torch.tensor(0., dtype=parameter.dtype)

    forward = first_derivative_forward(parameter[1], parameter[0], sample_time).unsqueeze(0)
    central = first_derivative_central(parameter[2:], parameter[:-2], sample_time)
    backward = first_derivative_backward(parameter[-1], parameter[-2], sample_time).unsqueeze(0)

    return torch.cat((forward, central, backward), dim=0)