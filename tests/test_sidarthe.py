import pytest
import torch

from torch_euler import Heun

dtype=torch.float64
train_size = 45

initial_params = {
        "alpha": 0.570,
        "beta": 0.0011,
        "gamma": 0.0011,
        "delta": 0.456,
        "epsilon": 0.171,
        "theta": 0.371,
        "xi": 0.125,
        "eta": 0.125,
        "mu": 0.012,
        "ni": 0.027,
        "tau": 0.003,
        "lambda": 0.034,
        "kappa": 0.017,
        "zeta": 0.017,
        "rho": 0.034,
        "sigma": 0.017
    }

init_cond = [0.9999681, 7.1e-06, 7.1e-06, 7.6e-06, 7.6e-06, 1.9e-06, 0.0, 6e-07]

params = {
        "alpha": [initial_params["alpha"]] * train_size,
        "beta": [initial_params["beta"]] * train_size,
        "gamma": [initial_params["gamma"]] * train_size,
        "delta": [initial_params["delta"]] * train_size,
        "epsilon": [initial_params["epsilon"]] * train_size,
        "theta": [initial_params["theta"]] * train_size,
        "xi": [initial_params["xi"]] * train_size,
        "eta": [initial_params["eta"]] * train_size,
        "mu": [initial_params["mu"]] * train_size,
        "ni": [initial_params["ni"]] * train_size,
        "tau": [initial_params["tau"]] * train_size,
        "lambda": [initial_params["lambda"]] * train_size,
        "kappa": [initial_params["kappa"]] * train_size,
        "zeta": [initial_params["zeta"]] * train_size,
        "rho": [initial_params["rho"]] * train_size,
        "sigma": [initial_params["sigma"]] * train_size
}

params = {key: torch.tensor(value, dtype=dtype) for key, value in
                        params.items()}

population = 1

def differential_equations(t, x):
    """
    Returns the right-hand side of SIDARTHE model
    :param t: time t at which right-hand side is computed
    :param x: state of model at time t
        x[0] = S
        x[1] = I
        x[2] = D
        x[3] = A
        x[4] = R
        x[5] = T
        x[6] = H
        x[7] = E
    :return: right-hand side of SIDARTHE model, i.e. f(t,x(t))
    """

    t = t.long()

    def get_param_at_t(param, _t):
        if 0 <= _t < param.shape[0]:
            return param[_t].unsqueeze(0)
        else:
            return param[-1].unsqueeze(0)

    # region parameters
    alpha = get_param_at_t(params["alpha"], t) / population
    beta = get_param_at_t(params["beta"], t) / population
    gamma = get_param_at_t(params["gamma"], t) / population
    delta = get_param_at_t(params["delta"], t) / population

    epsilon = get_param_at_t(params["epsilon"], t)
    theta = get_param_at_t(params["theta"], t)
    xi = get_param_at_t(params["xi"], t)
    eta = get_param_at_t(params["eta"], t)
    mu = get_param_at_t(params["mu"], t)
    ni = get_param_at_t(params["ni"], t)
    tau = get_param_at_t(params["tau"], t)
    lambda_ = get_param_at_t(params["lambda"], t)
    kappa = get_param_at_t(params["kappa"], t)
    zeta = get_param_at_t(params["zeta"], t)
    rho = get_param_at_t(params["rho"], t)
    sigma = get_param_at_t(params["sigma"], t)
    # endregion parameters

    S = x[0]
    I = x[1]
    D = x[2]
    A = x[3]
    R = x[4]
    T = x[5]
    H = x[6]
    E = x[7]

    # region equations

    S_dot = -S * (alpha * I + beta * D + gamma * A + delta * R)
    I_dot = -S_dot - (epsilon + xi + lambda_) * I
    D_dot = epsilon * I - (eta + rho) * D
    A_dot = xi * I - (theta + mu + kappa) * A
    R_dot = eta * D + theta * A - (ni + zeta) * R
    T_dot = mu * A + ni * R - (sigma + tau) * T
    H_dot = lambda_ * I + rho * D + kappa * A + zeta * R + sigma * T
    E_dot = tau * T

    # endregion equations

    return torch.cat((
        S_dot,
        I_dot,
        D_dot,
        A_dot,
        R_dot,
        T_dot,
        H_dot,
        E_dot
    ), dim=0)

def omega(t):
    if t >= 0:
        return torch.tensor([init_cond], dtype=dtype)
    else:
        return torch.tensor([[1.] + [0.] * 5], dtype=dtype)


t_grid = torch.linspace(0, 100, 100)

sol = Heun(differential_equations, omega, t_grid)

s = sol[:, 0]
i = sol[:, 1]
d = sol[:, 2]
a = sol[:, 3]
r = sol[:, 4]
t = sol[:, 5]
h = sol[:, 6]
e = sol[:, 7]

e_simple = init_cond[7] + torch.cumsum(t, dim=0)
h_simple = 1 - (s + i + d + a + r + t + e)

e_dist = torch.dist(e, e_simple)
h_dist = torch.dist(h, h_simple)
print(f"Distance between e and e_simple: {e_dist}")
print(f"Distance between h and h_simple: {h_dist}")
zero = torch.zeros(1, dtype=dtype)


def test_e():
    assert(torch.isclose(e_dist, zero))

def test_h():
    assert(torch.isclose(h_dist, zero))



