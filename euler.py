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


class EulerMethod:
    def __init__(self, f, x0, t_vals):
        """
        :param f: right hand side of Differential Equation dot{x} = f(t, x)
        :param x0: initial values
        :param t_vals: values on which to evaluate f (from t1 to T)
        """
        self.f = f
        self.x0 = x0
        self.t_vals = t_vals
        self.x_vals = np.zeros((len(self.t_vals), len(x0)))
        self.x_vals[0] = x0

    # t_vals = [0:0,1:1,2:2, ...]
    # x_vals = [0:a,1:b,2:c, ...]

    def run(self):
        for i in range(0, len(self.t_vals) - 1):
            t = self.t_vals[i]
            x_t = self.x_vals[i]
            epsilon = self.t_vals[i + 1] - t
            f_t = np.asarray(self.f(t, x_t))
            self.x_vals[i + 1] = x_t + epsilon * f_t

        return list(self.x_vals)


class DelayedEulerMethod():
    def __init__(self, f, omega, t_vals):
        self.f = f
        x0 = omega(0)
        x_vals = np.zeros((len(t_vals), len(x0)))
        x_vals[0] = x0
        self.t_series = TimeSeries(t_vals, x_vals, omega)

    def run(self):
        for i in range(0, len(self.t_series.time_grid) - 1):
            t = self.t_series.time_grid[i]
            x_t = self.t_series.values[i]
            epsilon = self.t_series.time_grid[i + 1] - t
            f_t = np.asarray(self.f(t, self.t_series, epsilon))
            self.t_series.values[i + 1] = x_t + epsilon * f_t

        return list(self.t_series.values)


gamma = 0.1
beta = 0.170
epsilon_s = 1e-6
S0 = 1 - epsilon_s
I0 = epsilon_s
ND = 500
TS = 0.01
tau = 6


def f(t, x):
    return [
        - beta * x[0] * x[1],
        beta * x[0] * x[1] - gamma * x[1],
        gamma * x[1]
    ]


def f_past_2(t, x, x_past, dt):
    return [
        - beta * x[0] * x[1],
        beta * x[0] * x[1] - gamma * x_past[1],
        gamma * x_past[1]
    ]


def f_past(t, X, dt):
    out = [
        - beta * X(t)[0] * X(t)[1],
        beta * X(t)[0] * X(t)[1] - gamma * X(t-tau)[1],
        gamma * X(t-tau)[1]
    ]

    if out[1] * dt + X(t)[1] < 0:
        out[1] = -X(t)[1] / dt
        out[2] = out[0] + X(t)[1] / dt

    if out[1] * dt + X(t)[1] > 1:
        out[1] = (1 - X(t)[1]) / dt
        out[2] = out[0] + (1 - X(t)[1]) / dt

    return out


def omega(t):
    return [
        0 if t < 0 else S0,
        0 if t < 0 else I0,
        0
    ]


if __name__ == '__main__':
    t_range = np.arange(0, ND, TS)
    euler = DelayedEulerMethod(f_past, omega, t_range)
    sol = euler.run()

    a = plt.figure(1)
    plt.plot(t_range, sol)
    plt.grid()
    a.show()
    print(f"x: {sol[len(sol) - 1][0]}, y: {sol[len(sol) - 1][2]}")
