import torch
import numpy as np
import matplotlib.pyplot as plt

class TimeSeries:
    def __init__(self, time_grid, values, omega):
        """
        Constructs a time series

        :param time_grid: array starting with 0 and containing the times at which the values in param values are defined
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
            idx = np.searchsorted(self.time_grid, t)
            x = self.values[idx]

        return x


def euler(f, omega, time_grid):
    """

    :param f: function describing the differential equations
    :param omega: function returning values from -inf to 0 as a N-Dim tuple
    :param time_grid: 1-Dim tensor representing the time-grid on which to integrate
    :return: 1-Dim tensor the same size as time_grid with values computed on the time grid
    """

    y0 = tuple(torch.tensor([y0_]) for y0_ in omega(0))
    time_grid = time_grid.to(y0[0])
    values = [y0]

    t_series = TimeSeries(time_grid.detach().numpy(), values, omega)

    for i in range(0, time_grid.shape[0] - 1):
        t_i = time_grid[i]
        t_next = time_grid[i+1]
        y_i = values[i]

        dt = t_next - t_i
        dy = tuple(dt * f_t for f_t in f(t_i, t_series, dt))
        y_next = tuple(y0_ + dy_ for y0_, dy_ in zip(y_i, dy))
        values.append(y_next)

    return values


gamma = torch.tensor([0.1], requires_grad=True)
beta = torch.tensor([0.170], requires_grad=True)
epsilon_s = 1e-6
S0 = 1 - epsilon_s
I0 = epsilon_s
ND = 200
TS = 1
tau = 9


def omega(t):
    return  (
        1. if t < 0 else S0,
        0. if t < 0 else I0,
        0.
    )

def f_past(t, X, dt):

    X_t = X(t)
    X_tau = X(t-tau)

    out = [
        - beta * X_t[0] * X_t[1],
        beta * X_t[0] * X_t[1] - gamma * X_tau[1],
        gamma * X_tau[1]
    ]

    if out[1] * dt + X_t[1] < 0:
        out[1] = -X_t[1] / dt
        out[2] = out[0] + X_t[1] / dt

    if out[1] * dt + X_t[1] > 1:
        out[1] = (1 - X_t[1]) / dt
        out[2] = out[0] + (1 - X_t[1]) / dt

    return tuple(out)


if __name__ == '__main__':
    t_range = torch.arange(0, ND, TS)
    sol = euler(f_past, omega, t_range)
    #print(sol)

    for i in range(1, len(sol)):
        tens = sol[i]
        x = tens[0].backward(torch.ones(ND, 1), retain_graph=True)
        print(x)

    a = plt.figure(1)
    plt.plot(t_range.detach().numpy(), sol)
    plt.grid()
    a.show()
    print(f"x: {sol[len(sol) - 1][0]}, y: {sol[len(sol) - 1][2]}")