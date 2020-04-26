from torch_euler import euler
from torchdiffeq import odeint, odeint_adjoint
from scipy.integrate import odeint as scipy_odeint
import torch

from matplotlib import pyplot as plt


gamma = torch.tensor([0.1])
beta = torch.tensor([0.8])
population = 1
epsilon_s = 1e-6
S0 = 1 - epsilon_s
I0 = epsilon_s
ND = 300
TS = 0.01


def omega(t):
    return  (
        1. if t < 0 else S0,
        0. if t < 0 else I0,
        0.
    )

def f_odeint(X_t, t):

    beta_t = beta.item()
    gamma_t = gamma.item()

    return [
        - beta_t * X_t[0] * X_t[1],
        beta_t * X_t[0] * X_t[1] - gamma_t * X_t[1],
        gamma_t * X_t[1]
    ]

def f_euler(T, X, dt):
    X_t = X(T)

    #beta_t = beta
    #gamma_t = gamma

    return torch.cat((
        - beta * X_t[0] * X_t[1],
        beta * X_t[0] * X_t[1] - gamma * X_t[1],
        gamma * X_t[1]
    ), dim=0)


class Sir(torch.nn.Module):
    def forward(self, t, y):
        X_t = y
        beta_t = beta
        gamma_t = gamma

        return torch.cat((
            - beta_t * X_t[0] * X_t[1],
            beta_t * X_t[0] * X_t[1] - gamma_t * X_t[1],
            gamma_t * X_t[1]
        ), dim=0)


if __name__ == '__main__':

    t_range = torch.arange(0, ND+TS, TS, dtype=torch.float32)
    init_cond = torch.tensor(omega(0))

    print("solving with scipy odeint")
    sol_odeint = scipy_odeint(f_odeint, init_cond.numpy(), t_range.numpy())
    sol_odeint = torch.tensor(sol_odeint)

    print("solving with explicit euler")
    sol_euler = euler(f_euler, omega, t_range)

    print("Solving with torchdiffeq odeint")
    sol_tdiffeq = odeint(Sir(), init_cond, t_range)

    a = plt.figure()
    plt.plot(t_range.numpy(), sol_odeint[:,0].numpy(), label="scipy_odeint", linestyle='-')
    plt.plot(t_range.numpy(), sol_euler[:,0].detach().numpy(), label="euler odeint", linestyle='-.')
    plt.plot(t_range.numpy(), sol_tdiffeq[:,0].detach().numpy(), label="torchdiffeq odeint", linestyle='--')
    plt.legend()
    a.show()

    b = plt.figure()
    plt.plot(t_range.numpy(), sol_odeint[:,1].numpy(), label="scipy_odeint", linestyle='-')
    plt.plot(t_range.numpy(), sol_euler[:,1].detach().numpy(), label="euler odeint", linestyle='-.')
    plt.plot(t_range.numpy(), sol_tdiffeq[:,1].detach().numpy(), label="torchdiffeq odeint", linestyle='--')
    plt.legend()
    b.show()

    c = plt.figure()
    plt.plot(t_range.numpy(), sol_odeint[:,2].numpy(), label="scipy_odeint", linestyle='-')
    plt.plot(t_range.numpy(), sol_euler[:,2].detach().numpy(), label="euler odeint", linestyle='-.')
    plt.plot(t_range.numpy(), sol_tdiffeq[:,2].detach().numpy(), label="torchdiffeq odeint", linestyle='--')
    plt.legend()
    c.show()

    max_odeint = max(sol_odeint[:,1])
    max_euler = max(sol_euler[:,1])

    print(max_euler - max_odeint)
