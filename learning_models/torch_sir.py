import time

import numpy
import torch
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch_euler import euler
from torchdiffeq import odeint

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as pl

class SirOptimizer(Optimizer):
    def __init__(self, params, etas, alpha, a, b, sample_time, momentum=True):
        """

        :param params: iterable containing parameters to be optimized
        :param etas: iterable containing learning rates in the same order as the parameter
        :param kwargs: keyword argument required:
            * eta_b, eta_g, eta_d : learning rates for beta, gamma,delta
            * a, b : parameters for learning rate decay
            * alpha: parameter for momentum term
        """

        self.etas = etas
        self.b = b
        self.a = a
        self.alpha = alpha
        self.momentum = momentum
        self.sample_time = sample_time
        defaults = dict()

        super(SirOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None,
        if closure is not None:
            loss = closure()

        for group in self.param_groups: # modded learning rates
            for idx, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue
                d_p = parameter.grad.data
                if self.momentum:
                    times = torch.arange(group["params"][0].shape[0], dtype=torch.float32)
                    times = times * self.sample_time #added AB
                    mu = torch.sigmoid(self.alpha * times)
                    eta_mod = self.a / (self.a + self.b * times)
                    etas = torch.tensor(self.etas)
                    etas = etas.unsqueeze(1) * eta_mod.unsqueeze(0)
                    update = [-etas[idx][0] * d_p[0]]
                    for t in range(1, d_p.size(0)):
                        momentum_term = -etas[idx][t] * d_p[t] + mu[t] * update[t - 1]
                        update.append(momentum_term)
                    parameter.data.add_(torch.tensor(update))
                else:
                    parameter.data.add_(-self.etas[idx] * d_p)


class SirEq:
    def __init__(self, beta, gamma, delta, population, init_cond, mode="dynamic", **kwargs):
        self.beta = torch.tensor(beta, requires_grad=True)
        self.gamma = torch.tensor(gamma, requires_grad=True)
        self.delta = torch.tensor(delta, requires_grad=True)

        self.population = population
        self.init_cond = init_cond

        self.b_reg = kwargs.get("b_reg", 1e4)
        self.c_reg = kwargs.get("c_reg", 1e4)
        self.d_reg = kwargs.get("d_reg", 1e4)
        self.bc_reg = kwargs.get("bc_reg", 1e4)
        self.der_1st_reg = kwargs.get("der_1st_reg", 1e3)
        self.der_2nd_reg = kwargs.get("der_2nd_reg", 1e3)
        self.sample_time = kwargs.get("sample_time", 1)

        if mode == "dynamic":
            self.diff_eqs = self.dynamic_diff_eqs
        else:
            self.diff_eqs = self.static_diff_eqs

    def omega(self, t):
        if t >= 0:
            return self.init_cond
        else:
            return [1, 0, 0]

    def dynamic_diff_eqs(self, T, X, dt):
        X_t = X(T)
        t = (T / self.sample_time).round().long()

        if t < self.beta.shape[0]:
            beta = self.beta[t] / self.population
            gamma = self.gamma[t]
        else:
            beta = self.beta[-1] / self.population
            gamma = self.gamma[-1]

        beta = beta.unsqueeze(0)
        gamma = gamma.unsqueeze(0)

        return torch.cat((
            - beta * X_t[0] * X_t[1],
            beta * X_t[0] * X_t[1] - gamma * X_t[1],
            gamma * X_t[1]
        ), dim=0)

    def static_diff_eqs(self, T, X, dt):
        X_t = X(T)
        t = T.long()

        beta = self.beta
        gamma = self.gamma

        return [
            - beta * X_t[0] * X_t[1],
            beta * X_t[0] * X_t[1] - gamma * X_t[1],
            gamma * X_t[1]
        ]

    # define first derivative losses
    @staticmethod
    def __first_derivative_central(f_x_plus_h, f_x_minus_h, h):
        return 0.5 * torch.pow((f_x_plus_h - f_x_minus_h) / (2 * h), 2)

    @staticmethod
    def __first_derivative_forward(f_x_plus_h, f_x, h):
        return 0.5 * torch.pow((f_x_plus_h - f_x) / h, 2)

    @staticmethod
    def __first_derivative_backward(f_x, f_x_minus_h, h):
        return 0.5 * torch.pow((f_x - f_x_minus_h) / h, 2)

    def __first_derivative_loss(self, parameter):
        sample_time = self.sample_time
        forward = self.__first_derivative_forward(parameter[1], parameter[0], sample_time).unsqueeze(0)
        central = self.__first_derivative_central(parameter[2:], parameter[:-2], sample_time)
        backward = self.__first_derivative_backward(parameter[-1], parameter[-2], sample_time).unsqueeze(0)

        t_grid = torch.arange(central.shape[0], dtype=torch.float32)
        return central #* (torch.pow(t_grid, 3) + 1.)
        #return central * (torch.pow(t_grid, 3) + 1.)
        #return torch.cat((forward, central, backward), dim=0)

    def first_derivative_loss(self):
        if self.der_1st_reg != 0:
            loss_1st_derivative_beta = self.__first_derivative_loss(self.beta)
            loss_1st_derivative_gamma = self.__first_derivative_loss(self.gamma)
            loss_1st_derivative_delta = self.__first_derivative_loss(self.delta)
            loss_1st_derivative_total = (
                    loss_1st_derivative_beta + loss_1st_derivative_gamma + loss_1st_derivative_delta
            )

            return self.der_1st_reg * torch.mean(loss_1st_derivative_total)
        else:
            return torch.zeros(1)

    # define second derivative losses
    @staticmethod
    def __second_derivative_central(f_x_plus_h, f_x, f_x_minus_h, h):
        return (f_x_plus_h - 2 * f_x + f_x_minus_h) / (h ** 2)

    @staticmethod
    def __second_derivative_forward(f_x_plus_2h, f_x_plus_h, f_x, h):
        return (f_x_plus_2h - 2 * f_x_plus_h + f_x) / (h ** 2)

    @staticmethod
    def __second_derivative_backward(f_x, f_x_minus_h, f_x_minus_2h, h):
        return (f_x - 2 * f_x_minus_h + f_x_minus_2h) / (h ** 2)

    def __second_derivative_loss(self, parameter: torch.Tensor):
        sample_time = self.sample_time
        forward = self.__second_derivative_forward(parameter[2], parameter[1], parameter[0], sample_time).unsqueeze(0)
        central = self.__second_derivative_central(parameter[2:], parameter[1:-1], parameter[:-2], sample_time)
        backward = self.__second_derivative_backward(parameter[-1], parameter[-2], parameter[-3], sample_time).unsqueeze(0)
        return torch.cat((forward, central, backward), dim=0)

    def second_derivative_loss(self):
        if self.der_2nd_reg != 0:
            loss_2nd_derivative_beta = self.__second_derivative_loss(self.beta)
            loss_2nd_derivative_gamma = self.__second_derivative_loss(self.gamma)
            loss_2nd_derivative_delta = self.__second_derivative_loss(self.delta)
            loss_2nd_derivative_total = (
                    loss_2nd_derivative_beta + loss_2nd_derivative_gamma + loss_2nd_derivative_delta
            )
            return self.der_2nd_reg * torch.mean(loss_2nd_derivative_total)
        else:
            return torch.zeros(1)

    @staticmethod
    def __loss_gte_one(parameter):
        return torch.where(parameter.ge(1.), torch.ones(1), torch.zeros(1)) * parameter.abs()

    @staticmethod
    def __loss_lte_zero(parameter: torch.Tensor):
        return torch.where(parameter.le(0.), torch.ones(1), torch.zeros(1)) * parameter.abs()

    def loss(self, w_hat, w_target):
        if isinstance(w_target, numpy.ndarray):
            w_target = torch.tensor(w_target)
        w_target = w_target.to(w_hat.dtype)

        # compute mse loss
        mse_loss = 0.5 * torch.mean(torch.pow((w_hat - w_target), 2))

        # REGULARIZATION TO PREVENT b,c,d from going out of bounds
        loss_reg_beta = self.b_reg * (self.__loss_gte_one(self.beta) + self.__loss_lte_zero(self.beta))
        loss_reg_gamma = self.c_reg * (self.__loss_gte_one(self.gamma) + self.__loss_lte_zero(self.gamma))
        loss_reg_delta = self.d_reg * (self.__loss_gte_one(self.delta) + self.__loss_lte_zero(self.delta))

        # compute total loss
        total_loss = mse_loss + \
            loss_reg_beta + loss_reg_gamma + loss_reg_delta

        return mse_loss, torch.mean(total_loss)

    def inference(self, time_grid):
        time_grid = time_grid.to(dtype=torch.float32)
        sol = euler(self.diff_eqs, self.omega, time_grid)
        z_hat = sol[:, 2]

        delta = self.delta
        len_diff = z_hat.shape[0] - delta.shape[0]
        if len_diff > 0:
            delta = torch.cat((delta, delta[-1].expand(len_diff)))

        w_hat = delta * z_hat

        return w_hat, sol

    def plot_params_over_time(self):
        fig, ax = pl.subplots()
        pl.title("Beta, Gamma, Delta over time")
        pl.grid(True)
        ax.plot(self.beta.detach().numpy(), '-g', label="beta")
        ax.plot(self.gamma.detach().numpy(), '-r', label="gamma")
        ax.plot(self.delta.detach().numpy(), '-b', label="delta")
        ax.margins(0.05)
        ax.legend()
        return fig

    def plot_sir_fit(self, w_hat, w_target):
        fig = pl.figure()
        pl.grid(True)
        pl.title("Estimated Deaths on fit")
        pl.plot(w_hat.detach().numpy(), '-', label='Estimated Deaths')
        pl.plot(w_target.detach().numpy(), '.r', label='Actual Deaths')
        pl.xlabel('Time in days')
        pl.ylabel('Deaths')
        return fig

    def params(self):
        return [self.beta, self.gamma, self.delta]

    @staticmethod
    def train(target, y0, z0, **params):
        beta = params["beta"]
        gamma = params["gamma"]
        delta = params["delta"]
        population = params["population"]
        t_start = params["t_start"]
        t_end = params["t_end"]
        b_reg = params.get("b_reg", 1e7)
        c_reg = params.get("c_reg", 1e7)
        d_reg = params.get("d_reg", 1e7)
        bc_reg = params.get("bc_reg", 1e7)
        der_1st_reg = params.get("der_1st_reg", 1e3)
        der_2nd_reg = params.get("der_2nd_reg", 1e3)
        momentum = params.get("momentum", True)
        n_epochs = params.get("n_epochs", 2000)
        t_inc = params.get("t_inc", 1)
        run_name = params.get("run_name", "test")
        lr_b, lr_g, lr_d = params["lr_b"], params["lr_g"], params["lr_d"]

        w_target = torch.tensor(target[t_start:t_end])
        time_grid = torch.arange(t_start, t_end + t_inc, t_inc)

        # init parameters
        epsilon = y0 / population
        epsilon_z = z0 / population
        S0 = 1 - (epsilon + epsilon_z)
        I0 = epsilon
        Z0 = epsilon_z
        S0 = S0 * population
        I0 = I0 * population
        Z0 = w_target[0].item()
        init_cond = (S0, I0, Z0)  # initialization of SIR parameters (Suscettible, Infected, Recovered)
        sir = SirEq(beta, gamma, delta, population, init_cond,
                    b_reg=b_reg, c_reg=c_reg, d_reg=d_reg,
                    der_1st_reg=der_1st_reg, der_2nd_reg=der_2nd_reg,
                    sample_time=t_inc
                    )

        # early stopping stuff
        best = 1e12
        thresh = 1e-5
        patience, n_lr_updts, max_no_improve, max_n_lr_updts = 0, 0, 75, 20
        best_beta, best_gamma, best_delta = sir.beta, sir.gamma, sir.delta

        optimizer = SirOptimizer(sir.params(), [lr_b, lr_g, lr_d], alpha=1 / 10, a=1.0, b=0.05, sample_time=t_inc, momentum=momentum)
        der_optimizer = SGD(sir.params(), lr=0.1)
        summary = SummaryWriter(f"runs/{run_name}")
        time_start = time.time()
        log_epoch_steps = 10

        # add initial params
        summary.add_figure("params_over_time", sir.plot_params_over_time(), close=True, global_step=-1)

        mse_losses = []
        for i in range(n_epochs):
            w_hat, _ = sir.inference(time_grid)
            w_hat = w_hat[slice(t_start, int(t_end / t_inc), int(1 / t_inc))]
            optimizer.zero_grad()

            # pure mse and mse + 0-1 boundary regularization
            mse_loss, reg_mse_loss = sir.loss(w_hat, w_target)

            #derivatives losses
            der_1st_loss = sir.first_derivative_loss()
            der_2nd_loss = sir.second_derivative_loss()

            #total loss
            total_loss = reg_mse_loss + der_1st_loss
            #total_loss = der_1st_loss

            retain_graph = (i % log_epoch_steps == 0)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sir.params(), 7.)
            """torch.nn.utils.clip_grad_norm_(sir.beta, 10.)
            torch.nn.utils.clip_grad_norm_(sir.gamma, 10.)
            torch.nn.utils.clip_grad_norm_(sir.delta, 10.)"""
            optimizer.step()

            if i % log_epoch_steps == 0:
                print(f"epoch {i} / {n_epochs}")
                # add current plot of params
                fig = sir.plot_params_over_time()
                summary.add_figure("params_over_time", fig, close=True, global_step=i)
                # add current fit
                fig = sir.plot_sir_fit(w_hat, w_target)
                summary.add_figure("sir fit", fig, close=True, global_step=i)

                mse_losses.append(mse_loss.detach().clone().numpy())
                summary.add_scalar("losses/mse_loss", mse_loss, global_step=i)
                summary.add_scalar("losses/tot_loss", total_loss, global_step=i)
                summary.add_scalar("losses/der_1st_loss", der_1st_loss, global_step=i)
                summary.add_scalar("losses/der_2nd_loss", der_2nd_loss, global_step=i)
                summary.add_scalar("losses/beta_grad_norm", sir.beta.grad.detach().clone().norm(), global_step=i)
                summary.add_scalar("losses/gamma_grad_norm", sir.gamma.grad.detach().clone().norm(), global_step=i)
                summary.add_scalar("losses/delta_grad_norm", sir.delta.grad.detach().clone().norm(), global_step=i)

                """optimizer.zero_grad()
                sir.loss(w_hat, w_target)[1].backward()
                scalars = {
                    'beta': sir.beta.grad.detach().clone().norm(),
                    'gamma': sir.gamma.grad.detach().clone().norm(),
                    'delta': sir.delta.grad.detach().clone().norm()
                }
                summary.add_scalars("grads/tot_loss", scalars, global_step=i)

                optimizer.zero_grad()
                sir.first_derivative_loss().backward()
                scalars = {
                    'beta': sir.beta.grad.detach().clone().norm(),
                    'gamma': sir.gamma.grad.detach().clone().norm(),
                    'delta': sir.delta.grad.detach().clone().norm()
                }
                summary.add_scalars("grads/der_1st_loss", scalars, global_step=i)"""

                """print("Loss at step %d: %.7f" % (i, mse_loss))
                print("beta: " + str(sir.beta.grad))
                print("gamma: " + str(sir.gamma.grad))
                print("delta: " + str(sir.delta.grad))"""
                time_step = time.time() - time_start
                time_start = time.time()
                print(f"Average time for epoch: {time_step / log_epoch_steps}\n")

            if mse_loss + thresh < best:
                # maintains the best solution found so far
                best = mse_loss
                best_beta = sir.beta
                best_gamma = sir.gamma
                best_delta = sir.delta
                patience = 0
            """elif patience < max_no_improve:
                patience += 1
            elif n_lr_updts < max_n_lr_updts:
                # when patience is over reduce learning rate by 2
                print("Reducing learning rate at step: %d" % i)
                lr_b, lr_g, lr_d = lr_b / 2, lr_g / 2, lr_d / 2
                optimizer.etas = [lr_b, lr_g, lr_d]
                n_lr_updts += 1
                patience = 0
            else:
                # after too many reductions early stops
                print("Early stop at step: %d" % i)
                break"""

        print("Best: " + str(best))
        print(best_beta)
        print(best_gamma)
        print(best_delta)
        print("\n")

        return sir, mse_losses
