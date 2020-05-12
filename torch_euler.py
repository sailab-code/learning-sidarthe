import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim


class TimeSeries:
    def __init__(self, time_grid, values, omega):

        """
        Constructs a time series

        :param time_grid: array starting with 0 and containing the times at which the values in param values are defined (don't use time increase with odd exponent, e.g 1e-1 is wrong)
        :param values: values assumed by the time series
        :param omega: callable in the form of f(t) which returns the values for t < 0
        """
        self.time_grid = time_grid
        self.values = values
        self.omega = omega

    def __call__(self, t):
        """
        Gets the value at time t

        :param t: time at which to get values
        :return: the value at time t
        """
        if t < 0:
            x = self.omega(t)
        else:
            x = self.values[self.time_grid[:self.values.shape[0]] == t][0]

        return x


def euler(f, omega, time_grid):
    """

    :param f: function describing the differential equations
    :param omega: function returning values from -inf to 0 as a N-Dim tuple
    :param time_grid: 1-Dim tensor representing the time-grid on which to integrate
    :return: 1-Dim tensor the same size as time_grid with values computed on the time grid
    """

    y0 = torch.tensor([omega(0)], requires_grad=True)
    time_grid = time_grid.to(y0[0])
    values = y0.clone()

    for i in range(0, time_grid.shape[0] - 1):
        t_i = time_grid[i]
        t_next = time_grid[i+1]
        y_i = values[i]
        dt = torch.tensor([t_next - t_i])
        dy = f(t_i, y_i) * dt
        y_next = y_i + dy
        y_next = y_next.unsqueeze(0)
        values = torch.cat((values, y_next), dim=0)

    return values

def Heun(f, omega, time_grid):
    """

    Heun's method
    :param f: function describing the differential equations
    :param omega: function returning values from -inf to 0 as a N-Dim tuple
    :param time_grid: 1-Dim tensor representing the time-grid on which to integrate
    :return: 1-Dim tensor the same size as time_grid with values computed on the time grid

    NOTE: not expected to reach second-order accuracy if dt is variable
    """

    y0 = torch.tensor([omega(0)])
    time_grid = time_grid.to(y0[0])
    values = y0

    for i in range(0, time_grid.shape[0] - 1):
        t_i = time_grid[i]
        t_next = time_grid[i+1]
        y_i = values[i]
        dt = torch.tensor([t_next - t_i])
        f1 = f(t_i, y_i)
        f2 = f(t_next, y_i + dt * f1)
        dy = 0.5 * dt * (f1 + f2)
        y_next = y_i + dy
        y_next = y_next.unsqueeze(0)
        values = torch.cat((values, y_next), dim=0)

    return values

def RK4(f, omega, time_grid):
    """

    Fourth order explicit Runge-Kutta method
    :param f: function describing the differential equations
    :param omega: function returning values from -inf to 0 as a N-Dim tuple
    :param time_grid: 1-Dim tensor representing the time-grid on which to integrate
    :return: 1-Dim tensor the same size as time_grid with values computed on the time grid

    NOTE: not expected to reach second-order accuracy if dt is variable
    """

    y0 = torch.tensor([omega(0)])
    time_grid = time_grid.to(y0[0])
    values = y0

    for i in range(0, time_grid.shape[0] - 1):
        t_i = time_grid[i]
        t_next = time_grid[i+1]
        y_i = values[i]
        dt = torch.tensor([t_next - t_i])
        dtd2 = 0.5 * dt
        f1 = f(t_i, y_i)
        f2 = f(t_i + dtd2, y_i + dtd2 * f1)
        f3 = f(t_i + dtd2, y_i + dtd2 * f2)
        f4 = f(t_next, y_i + dt * f3)
        dy = 1/6 * dt * (f1 + 2 * (f2 + f3) +f4)
        y_next = y_i + dy
        y_next = y_next.unsqueeze(0)
        values = torch.cat((values, y_next), dim=0)

    return values

N = 1
gamma = torch.tensor([0.3] * N, requires_grad=True)
beta = torch.tensor([0.8] * N, requires_grad=True)
population = 1
epsilon_s = 1e-6
S0 = 1 - epsilon_s
I0 = epsilon_s
ND = 200
TS = 1
tau = torch.tensor([1.], requires_grad=True)


def omega(t):
    return  (
        1. if t < 0 else S0,
        0. if t < 0 else I0,
        0.
    )

def dynamic_f(T, X):
    X_t = X
    t = T.long()

    if t < beta.shape[0]:
        beta_t = beta[t] / population
        gamma_t = gamma[t]
    else:
        beta_t = beta[-1] / population
        gamma_t = gamma[-1]

    return torch.cat((
        - beta * X_t[0] * X_t[1],
        beta * X_t[0] * X_t[1] - gamma * X_t[1],
        gamma * X_t[1]
    ), dim=0)        

    # temp = [
    #     - beta_t * X_t[0] * X_t[1],
    #     beta_t * X_t[0] * X_t[1] - gamma_t * X_t[1],
    #     gamma_t * X_t[1]
    # ]

    # out = torch.stack(tuple(f_t.unsqueeze(0) for f_t in temp), dim=1)

    # return out

def f_past(t, X, dt):

    X_t = X(t)
    X_tau = X(t-tau)

    out = [
        - beta[0] * X_t[0] * X_t[1],
        beta[0] * X_t[0] * X_t[1] - gamma[0] * X_tau[1],
        gamma[0] * X_tau[1]
    ]

    if out[1] * dt + X_t[1] < 0:
        out[1] = -X_t[1] / dt[0]
        out[2] = out[0] + X_t[1] / dt[0]

    if out[1] * dt + X_t[1] > 1:
        out[1] = (1 - X_t[1]) / dt
        out[2] = out[0] + (1 - X_t[1]) / dt

    return out


epochs = 251
lr = 1e-3
if __name__ == '__main__':
    t_range = torch.arange(0, ND, TS)

    optimizer = optim.SGD([beta, gamma], lr=lr, momentum=0.9)
    for epoch in range(0, epochs):
        print("epoch {}".format(epoch))
        optimizer.zero_grad()
        # sol = euler(dynamic_f, omega, t_range)
        # sol = Heun(dynamic_f, omega, t_range)
        sol = RK4(dynamic_f, omega, t_range)
        z_hat = sol[-1][2]

        z_target = torch.tensor([[0.6]])

        loss = torch.pow(z_target - z_hat, 2)
        #print(k)
        loss.backward()
        print(beta.grad)
        print(gamma.grad)
        print(tau.grad)
        #print(z_hat)
        optimizer.step()
        # update params

        if epoch % 50 == 0:
            a = plt.figure(1)
            plt.plot(t_range.detach().numpy(), sol.detach().numpy())
            plt.grid()
            a.show()

            print("loss: {}".format(loss))
            print("beta: {}".format(beta))
            print("gamma: {}".format(gamma))

    print("loss: {}".format(loss))
    print("beta: {}".format(beta))
    print("gamma: {}".format(gamma))
