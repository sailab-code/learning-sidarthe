import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, f, omega, t_vals, tau):
        self.tau = tau
        self.f = f
        self.omega = omega
        self.t_vals = t_vals
        x0 = self.omega(0)
        self.x_vals = np.zeros((len(self.t_vals), len(x0)))
        self.x_vals[0] = x0
        self.tau = tau

    def run(self):
        for i in range(0, len(self.t_vals) - 1):
            t = self.t_vals[i]
            x_t = self.x_vals[i]
            t_past = t - self.tau
            if t_past < 0:
                x_past = self.omega(t_past)
            else:
                idx = np.searchsorted(self.t_vals, t_past)
                x_past = self.x_vals[idx]
            epsilon = self.t_vals[i + 1] - t
            f_t = np.asarray(self.f(t, x_t, x_past, epsilon))
            self.x_vals[i + 1] = x_t + epsilon * f_t

        return list(self.x_vals)


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


def f_past(t, x, x_past, dt):
    out = [
        - beta * x[0] * x[1],
        beta * x[0] * x[1] - gamma * x_past[1],
        gamma * x_past[1]
    ]

    if out[1] * dt + x[1] < 0:
        out[1] = -x[1] / dt
        out[2] = out[0] + x[1] / dt

    if out[1] * dt + x[1] > 1:
        out[1] = (1 - x[1]) / dt
        out[2] = out[0] + (1 - x[1]) / dt

    return out


def omega(t):
    return [
        0 if t < 0 else S0,
        0 if t < 0 else I0,
        0
    ]


if __name__ == '__main__':
    t_range = np.arange(0, ND, TS)
    euler = DelayedEulerMethod(f_past, omega, t_range, tau)
    sol = euler.run()

    a = plt.figure(1)
    plt.plot(t_range, sol)
    plt.grid()
    a.show()
    print(f"x: {sol[len(sol)-1][0]}, y: {sol[len(sol)-1][2]}")
