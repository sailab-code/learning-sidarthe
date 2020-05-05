import torch
from torch.optim import Adam
from torch.optim import SGD
from matplotlib import pyplot as pl
import numpy as np


def first_derivative_central(f_x_plus_h, f_x_minus_h, h):
    return (f_x_plus_h - f_x_minus_h) / (2 * h)


def first_derivative_forward(f_x_plus_h, f_x, h):
    return (f_x_plus_h - f_x) / h


def first_derivative_backward(f_x, f_x_minus_h, h):
    return (f_x - f_x_minus_h) / h


def loss_derivative(parameter: torch.Tensor, sample_time=1.):
    forward = first_derivative_forward(parameter[1], parameter[0], sample_time).unsqueeze(0)
    central = first_derivative_central(parameter[2:], parameter[:-2], sample_time)
    backward = first_derivative_backward(parameter[-1], parameter[-2], sample_time).unsqueeze(0)
    return torch.cat((forward, central, backward), dim=0)


def second_derivative_central(f_x_plus_h, f_x, f_x_minus_h, h):
    return (f_x_plus_h - 2 * f_x + f_x_minus_h) / (h ** 2)


def second_derivative_forward(f_x_plus_2h, f_x_plus_h, f_x, h):
    return (f_x_plus_2h - 2 * f_x_plus_h + f_x) / (h ** 2)


def second_derivative_backward(f_x, f_x_minus_h, f_x_minus_2h, h):
    return (f_x - 2 * f_x_minus_h + f_x_minus_2h) / (h ** 2)


def second_derivative(parameter: torch.Tensor, sample_time):
    forward = second_derivative_forward(parameter[2], parameter[1], parameter[0], sample_time).unsqueeze(0)
    central = second_derivative_central(parameter[2:], parameter[1:-1], parameter[:-2], sample_time)
    backward = second_derivative_backward(parameter[-1], parameter[-2], parameter[-3], sample_time).unsqueeze(0)
    return torch.cat((forward, central, backward), dim=0)


sample_step = 0.1
epochs = 500000
lr = 0.00001

t_grid = torch.arange(0, 20, sample_step, dtype=torch.float)
#initial_u = torch.pow(t_grid, 2)
initial_u = torch.reciprocal(1 + torch.pow(t_grid, 2))
u = initial_u.clone().detach().requires_grad_(True)

"""
pl.figure()
pl.title("derivative u")
pl.subplot(211)
pl.grid(True)
pl.ylabel("1st der")
pl.xlabel("Time")
pl.plot(t_grid[:-1], first_derivative_forward(initial_u[1:], initial_u[:-1], sample_step))
pl.subplot(212)
pl.grid(True)
pl.ylabel("2nd der")
pl.xlabel("Time")
pl.plot(t_grid, second_derivative(initial_u, sample_step))
pl.show()
"""


#optimizer = Adam([u], lr=0.0001)
optimizer = SGD([u], lr=0.001)
for epoch in range(epochs):
    optimizer.zero_grad()
    #loss = torch.mean(loss_derivative(u, sample_step))
    loss = torch.mean(0.5 * torch.pow(first_derivative_forward(u[1:], u[:-1], sample_step), 2))
    print(f"loss at epoch {epoch}: {loss}")
    loss.backward()
    """
    der_2nd = second_derivative_central(u[2:], u[1:-1], u[:-2], sample_step)
    manual_grad = torch.cat((
        - first_derivative_forward(u[1], u[0], sample_step).unsqueeze(0) / sample_step,
        der_2nd,
        first_derivative_backward(u[-1], u[-2], sample_step).unsqueeze(0) / sample_step
    ), dim=0)
    """
    der_2nd = second_derivative(u, sample_step)
    u.grad.data = -der_2nd.data / u.shape[0]
    optimizer.step()


pl.figure()
pl.title("U curve")
pl.subplot(211)
pl.ylim(0.,1)
pl.grid(True)
pl.xlabel('Epochs')
pl.ylabel('initial u')
pl.plot(t_grid, initial_u.numpy(), '-g', label="initial u")
pl.subplot(212)
pl.xlabel('Epochs')
pl.ylabel('final u')
pl.grid(True)
pl.plot(t_grid, u.detach().numpy(), '-r', label="final u")
pl.show()

