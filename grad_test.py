import torch
from torch.optim import SGD

from torch_euler import euler

N = 10
sampling = 0.05
time_grid = torch.arange(0, N, sampling)
beta = torch.arange(0, N, sampling, dtype=torch.float32, requires_grad=True)


def f(T, X, dt):
    a = T * 1 / sampling
    t = (T * 1 / sampling).long()
    return [
        beta[t]
    ]

def omega(t):
    return [0.]


N_n = int(N / sampling)


def second_derivative_central(f_x_plus_h, f_x, f_x_minus_h):
    return f_x_plus_h - 2 * f_x + f_x_minus_h


def second_derivative_forward(f_x_plus_2h, f_x_plus_h, f_x):
    return f_x_plus_2h - 2 * f_x_plus_h + f_x


def second_derivative_backward(f_x, f_x_minus_h, f_x_minus_2h):
    return f_x - 2 * f_x_minus_h + f_x_minus_2h


def loss_second_derivative(parameter: torch.Tensor, t):
    if t == 0:
        l = second_derivative_forward(parameter[t + 2], parameter[t + 1], parameter[t])
    elif t < parameter.shape[0] - 1:
        l = second_derivative_central(parameter[t + 1], parameter[t], parameter[t - 1])
    else:
        l = second_derivative_backward(parameter[t], parameter[t - 1], parameter[t - 2])

    return l



def first_derivative_central(f_x_plus_h, f_x_minus_h, h):
    return torch.pow((f_x_plus_h - f_x_minus_h) / (2 * h), 2)


def first_derivative_forward(f_x_plus_h, f_x, h):
    return torch.pow((f_x_plus_h - f_x) / h, 2)


def first_derivative_backward(f_x, f_x_minus_h, h):
    return torch.pow((f_x - f_x_minus_h) / h, 2)


def loss_first_derivative(parameter: torch.Tensor, h):


    """
    par_left_shift = torch.roll(parameter, shifts=-1, dims=0)
    par_left_shift[-1] = torch.zeros(1)
    par_right_shift = torch.roll(parameter, shifts=1, dims=0)
    par_right_shift[-1] = torch.zeros(1)
    """

    forward = first_derivative_forward(parameter[1], parameter[0], h).unsqueeze(0)
    central = first_derivative_central(parameter[2:], parameter[:-2], h)
    backward = first_derivative_backward(parameter[-1], parameter[-2], h).unsqueeze(0)

    return torch.cat((forward, central, backward), dim=0)


lr = 0.1
n_epochs = 4000
debug_interval = slice(0, 5)
if __name__ == '__main__':


    optimizer = SGD([beta], lr=lr)
    for epoch in range(n_epochs):
        print(f"epoch: {epoch}")
        print(f"beta: {beta[debug_interval]}")
        optimizer.zero_grad()
        sol = euler(f, omega, time_grid)

        loss_t_square = 0.5 * torch.pow(sol, 2).squeeze()
        #loss_t_derivative = loss_first_derivative(beta, sampling)
        loss_square = torch.mean(loss_t_square)
        #loss_derivative = torch.mean(loss_t_derivative)

        """if epoch % 50 == 0:
            print(f"loss square: {loss_t_square[debug_interval]}")
            print(f"loss derivative: {loss_t_derivative[debug_interval]}")
            loss_square.backward(retain_graph=True)
            print(f"square grad: {beta.grad[debug_interval]}")
            optimizer.zero_grad()

            loss_derivative.backward(retain_graph=True)
            print(f"derivative grad: {beta.grad[debug_interval]}")
            optimizer.zero_grad()
        """

        loss = loss_square # + sampling * loss_derivative
        loss.backward()
        torch.nn.utils.clip_grad_norm_([beta], 0.1)
        optimizer.step()

        print(f"loss: {loss}")
        print("\n\n")




        """
        loss_d = torch.zeros_like(beta)
        for h in range(0, beta.shape[0]):
            for i in range(h, beta.shape[0] - 1):
                loss_d[h] = loss_d[h] + sol[i].detach()

        loss_d = torch.zeros_like(beta)
        for h in range(0, beta.shape[0]):
            for i in range(h, beta.shape[0] - 1 ):
                loss_d[h] = loss_d[h] + sol[i + 1].detach()

        #print(loss)

        new_beta = torch.zeros(N_n)

        x = torch.ones(N_n, 1)
        #loss.backward(x, retain_graph=True)
        loss.backward()

        print(f"soluzione: {sol.flatten()}")
        print(f"loss_derivata_manuale: {loss_d.flatten()}")
        print(f"loss_autograd: {beta.grad.flatten() * 10.}")
        """

    print(beta)
    sol = euler(f, omega, time_grid)

    loss = torch.mean(0.5 * torch.pow(sol, 2))
    print(loss)

    """
    for i in range(0, N_n):
        if beta.grad is not None:
            beta.grad.data = torch.zeros(N_n)
        x = torch.zeros(N_n, 1)
        x[i][0] = 1.
        loss.backward(x, retain_graph=True)
        #print(beta.grad)
    """

    print("ok")

"""
x0 = 0
x1 = beta[1] + x0
x2 = beta[2] + x1 = beta[2] + beta[1] + x0

x0 + 2*beta[1] + beta[2]
"""
