import time

import numpy
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
from torch_euler import euler, Heun
from torchdiffeq import odeint

import matplotlib.pyplot as pl

from utils.visualization_utils import generic_plot, format_xtick, Curve


class SirOptimizer(Optimizer):
    def __init__(self, params, etas, m, a, b, momentum=True):
        """

        :param params: iterable containing parameters to be optimized
        :param etas: iterable containing learning rates in the same order as the parameter
        :param kwargs: keyword argument required:
            * eta_b, eta_g, eta_d : learning rates for beta, gamma,delta
            * a, b : parameters for learning rate decay
            * alpha: parameter for momentum term
        """

        self.etas = etas
        self.a = a
        self.b = b
        self.m = m
        self.momentum = momentum
        defaults = dict()

        super(SirOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None,
        if closure is not None:
            loss = closure()

        for group in self.param_groups:  # modded learning rates
            for idx, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                # # print(p.grad)
                # d_p = parameter.grad.data
                # lr_t = -etas[idx] * d_p
                # mu_t = torch.cumprod(torch.flip(torch.cat((torch.ones(1), mu[:-1])), dims=[0]), dim=0)  # 1, mu[0], mu[0]*mu[1], ...
                # update = torch.cumsum(lr_t * mu_t, dim=0)

                d_p = parameter.grad.data
                if self.momentum:
                    times = torch.arange(group["params"][0].shape[0], dtype=torch.float32)
                    # times = times * self.sample_time #added AB
                    mu = torch.sigmoid(self.m * times)
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

        self.b_reg = kwargs.get("b_reg", 1e5)
        self.c_reg = kwargs.get("c_reg", 1e5)
        self.d_reg = kwargs.get("d_reg", 1e9)
        self.bc_reg = kwargs.get("bc_reg", 1e5)
        self.der_1st_reg = kwargs.get("der_1st_reg", -1)
        self.der_2nd_reg = kwargs.get("der_2nd_reg", -1)
        self.use_alpha = kwargs.get("use_alpha", False)
        self.integrator = kwargs.get("integrator", euler)
        input_size = kwargs.get("mlp_input", 4)
        hidden_size = kwargs.get("mlp_hidden", 6)
        self.init_alpha(self.use_alpha, input_size=input_size, hidden_size=hidden_size)

        self.sample_time = kwargs.get("sample_time", 1)

        self.y_loss_weight = kwargs.get("y_loss_weight", 0.0)

        if mode == "dynamic":
            self.diff_eqs = self.dynamic_diff_eqs
        else:
            self.diff_eqs = self.static_diff_eqs

    def get_policy_code(self, t):
        policy_code = torch.zeros(4)
        if 3 < t <= 13:
            policy_code[0] = 1.0
        elif 13 < t <= 27:
            policy_code[1] = 1.0
        elif t > 27:
            policy_code[2] = 1.0

        return policy_code

    def omega(self, t):
        if t >= 0:
            return self.init_cond
        else:
            return [1, 0]

    def dynamic_diff_eqs(self, T, X):
        X_t = X
        t = T.long()

        policy_code = self.get_policy_code(t)
        alpha = self.alpha(policy_code)

        if 0 < t < self.beta.shape[0]:
            beta = (self.beta[t] / self.population) * alpha
            # gamma = self.gamma[t]
            gamma = self.gamma[-1]
        else:
            beta = (self.beta[-1] / self.population) * alpha
            gamma = self.gamma[-1]

        beta = beta.unsqueeze(0)
        gamma = gamma.unsqueeze(0)

        return torch.cat((
            - beta * X_t[0] * X_t[1],
            beta * X_t[0] * X_t[1] - gamma * X_t[1]
        ), dim=0)

    def static_diff_eqs(self, T, X):
        X_t = X
        t = T.long()

        policy_code = self.get_policy_code(t)

        beta = self.alpha(policy_code) * self.beta / self.population
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

    def init_alpha(self, use_alpha, input_size, hidden_size, output_size=1):
        """mlp"""

        if use_alpha:
            self.nn_alpha = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size),
                                          nn.Linear(in_features=hidden_size, out_features=output_size),
                                          nn.Sigmoid())

            self.alpha = self.net_alpha
        else:
            self.alpha = self._constant_alpha

    def net_alpha(self, policy):
        alpha = self.nn_alpha(policy).squeeze()
        return torch.where(torch.gt(policy.sum(), 0), alpha, torch.tensor(1.0))

    def _constant_alpha(self, policy):
        return torch.tensor(1.0)

    def mape(self, w_hat, w_target):
        if isinstance(w_target, numpy.ndarray) or isinstance(w_target, list):
            w_target = torch.tensor(w_target, dtype=w_hat.dtype)
        return torch.mean(torch.abs((w_hat - w_target) / w_target))

    def __first_derivative_loss(self, parameter):
        if parameter.shape[0] < 3: # must have at least 3 values to properly compute first derivative
            return 0.
        sample_time = self.sample_time
        forward = self.__first_derivative_forward(parameter[1], parameter[0], sample_time).unsqueeze(0)
        central = self.__first_derivative_central(parameter[2:], parameter[:-2], sample_time)
        backward = self.__first_derivative_backward(parameter[-1], parameter[-2], sample_time).unsqueeze(0)

        return torch.cat((forward, central, backward), dim=0)
        # return central

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
        if parameter.shape[0] < 3: # must have at least 3 values to properly compute first derivative
            return 0.

        sample_time = self.sample_time
        forward = self.__second_derivative_forward(parameter[2], parameter[1], parameter[0], sample_time).unsqueeze(0)
        central = self.__second_derivative_central(parameter[2:], parameter[1:-1], parameter[:-2], sample_time)
        backward = self.__second_derivative_backward(parameter[-1], parameter[-2], parameter[-3],
                                                     sample_time).unsqueeze(0)
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

    def loss(self, w_hat, w_target, y_hat=None, y_target=None):
        if isinstance(w_target, numpy.ndarray) or isinstance(w_target, list):
            w_target = torch.tensor(w_target, dtype=w_hat.dtype)
            y_target = torch.tensor(y_target, dtype=w_hat.dtype)

        w_mse_loss = 0.5 * torch.mean(torch.pow((w_hat - w_target), 2))

        if y_hat is not None and y_target is not None:
            y_mse_loss = 0.5 * torch.mean(torch.pow((y_hat - y_target), 2))
        else:
            y_mse_loss = 0.0

        # REGULARIZATION TO PREVENT b,c,d from going out of bounds
        loss_reg_beta = self.b_reg * (self.__loss_gte_one(self.beta) + self.__loss_lte_zero(self.beta))
        loss_reg_gamma = self.c_reg * (self.__loss_gte_one(self.gamma) + self.__loss_lte_zero(self.gamma))
        loss_reg_delta = self.d_reg * (self.__loss_gte_one(self.delta) + self.__loss_lte_zero(self.delta))

        # compute total loss
        mse_loss = ((1.0 - self.y_loss_weight) * w_mse_loss) + self.y_loss_weight * y_mse_loss
        total_loss = mse_loss + \
            torch.mean(loss_reg_beta + loss_reg_gamma + loss_reg_delta)

        return mse_loss, w_mse_loss, y_mse_loss, total_loss

    def inference(self, time_grid):
        time_grid = time_grid.to(dtype=torch.float32)
        sol = self.integrator(self.diff_eqs, self.omega, time_grid)
        z_hat = self.population - sol[:, 0] - sol[:, 1]
        sol = torch.cat((sol, z_hat.unsqueeze(1)), dim=1)

        delta = self.delta
        len_diff = z_hat.shape[0] - delta.shape[0]
        if len_diff > 0:
            delta = torch.cat((delta, delta[-1].expand(len_diff)))

        w_hat = delta * z_hat

        return w_hat, sol[:, 1], sol

    def plot_params_over_time(self):
        # BETA, GAMMA, DELTA
        size = self.beta.shape[0]
        pl_x = list(range(size))  # list(range(len(beta)))
        beta_pl = Curve(pl_x, self.beta.detach().numpy(), '-g', "$\\beta$")
        gamma_pl = Curve(pl_x, [self.gamma.detach().numpy()] * size, '-r', "$\gamma$")
        delta_pl = Curve(pl_x, [self.delta.detach().numpy()] * size, '-b', "$\delta$")
        params_curves = [beta_pl, gamma_pl, delta_pl]

        if self.use_alpha:
            alpha = numpy.np.concatenate(
                [self.alpha(self.get_policy_code(t)).detach().numpy().reshape(1) for t in range(size)], axis=0)
            alpha_pl = Curve(pl_x, alpha, '-', "$\\alpha$")
            beta_alpha_pl = Curve(pl_x, alpha * self.beta.detach().numpy(), '-', "$\\alpha \cdot \\beta$")
            params_curves.append(alpha_pl)
            params_curves.append(beta_alpha_pl)

        bgd_pl_title = "beta, gamma, delta"
        return generic_plot(params_curves, bgd_pl_title, None, formatter=format_xtick)

    def plot_sir_fit(self, w_hat, w_target):
        fig = pl.figure()
        pl.grid(True)
        pl.title("Estimated Deaths on fit")
        pl.plot(w_hat.detach().numpy(), '-', label='Estimated Deaths')
        pl.plot(w_target, '.r', label='Actual Deaths')
        pl.xlabel('Time in days')
        pl.ylabel('Deaths')
        return fig

    def params(self):
        # return [self.beta, self.gamma, self.delta] + list(self.alpha.parameters())
        return [self.beta, self.gamma, self.delta]

    def set_params(self, beta, gamma, delta):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    @staticmethod
    def train(w_target, y_target, **params):
        beta = params["beta"]
        gamma = params["gamma"]
        delta = params["delta"]
        population = params["population"]
        t_start = params["t_start"]
        t_end = params["t_end"]

        n_epochs = params.get("n_epochs", 2000)
        momentum = params.get("momentum", True)

        b_reg = params.get("b_reg", 1e7)
        c_reg = params.get("c_reg", 1e7)
        d_reg = params.get("d_reg", 1e8)
        bc_reg = params.get("bc_reg", 1e7)

        der_1st_reg = params.get("der_1st_reg", 1e3)
        der_2nd_reg = params.get("der_2nd_reg", 1e3)

        summary = params.get("tensorboard", None)
        integrator = params.get("integrator", None)

        y_loss_weight = params.get("y_loss_weight", 0.0)
        use_alpha = params.get("use_alpha", True)
        val_size = params.get("val_size", 7)

        m = params.get("m", 1/9)
        a = params.get("a", 3.0)
        b = params.get("b", 0.05)

        lr_b, lr_g, lr_d, lr_a = params["lr_b"], params["lr_g"], params["lr_d"], params["lr_a"]

        t_inc = params.get("t_inc", 1)


        train_time_grid = torch.arange(t_start, t_end + t_inc, t_inc)
        train_target_slice = slice(t_start, t_end, 1)
        train_hat_slice = slice(int(t_start / t_inc), int(t_end / t_inc), int(1 / t_inc))

        val_time_grid = torch.arange(t_start, t_end + val_size + t_inc, t_inc)
        val_target_slice = slice(t_end, t_end + val_size, 1)
        val_hat_slice = slice(int(t_end / t_inc), int((t_end + val_size) / t_inc), int(1 / t_inc))


        train_w_target = torch.tensor(w_target[train_target_slice], dtype=torch.float32)
        train_y_target = torch.tensor(y_target[train_target_slice], dtype=torch.float32)

        val_w_target = torch.tensor(w_target[val_target_slice], dtype=torch.float32)
        val_y_target = torch.tensor(y_target[val_target_slice], dtype=torch.float32)

        # init parameters
        epsilon = train_y_target[t_start].item() / population
        epsilon_z = train_w_target[t_start].item() / population
        S0 = 1 - (epsilon + epsilon_z)
        I0 = epsilon
        S0 = S0 * population
        I0 = I0 * population
        Z0 = epsilon_z

        init_cond = (S0, I0)  # initialization of SIR parameters (Suscettible, Infected, Recovered)
        sir = SirEq(beta, gamma, delta, population, init_cond,
                    b_reg=b_reg, c_reg=c_reg, d_reg=d_reg, bc_reg=bc_reg,
                    der_1st_reg=der_1st_reg, der_2nd_reg=der_2nd_reg,
                    sample_time=t_inc, use_alpha=use_alpha, y_loss_weight=y_loss_weight,
                    integrator=integrator)

        # early stopping stuff
        best = 1e12
        best_epoch = -1
        thresh = 5e-5
        patience, n_lr_updts, max_no_improve, max_n_lr_updts = 0, 0, 25, 5
        best_beta, best_gamma, best_delta = sir.beta, sir.gamma, sir.delta

        # optimizer = SirOptimizer(sir.params(), [lr_b, lr_g, lr_d, lr_a, lr_a, lr_a, lr_a], alpha=1 / 7, a=3.0, b=0.05) # fixme assigning of lrs
        optimizer = SirOptimizer(sir.params(), [lr_b, lr_g, lr_d], m=m, a=a, b=b,
                                 momentum=momentum)
        if use_alpha:
            net_optimizer = Adam(sir.nn_alpha.parameters(), lr_a)

        time_start = time.time()
        mse_losses, der_1st_losses, der_2nd_losses = [], [], []



        log_epoch_steps = 50
        validation_epoch_steps = 10

        # add initial params
        if summary is not None:
            summary.add_figure("params_over_time", sir.plot_params_over_time(), close=True, global_step=-1)

        for i in range(n_epochs):
            w_hat, y_hat, _ = sir.inference(train_time_grid)
            w_hat = w_hat[train_hat_slice]
            y_hat = y_hat[train_hat_slice]
            optimizer.zero_grad()

            if use_alpha:
                net_optimizer.zero_grad()

            mse_loss, w_loss, y_loss, total_loss = sir.loss(w_hat, train_w_target, y_hat, train_y_target)

            # derivatives losses
            der_1st_loss = sir.first_derivative_loss()
            der_2nd_loss = sir.second_derivative_loss()

            total_loss = torch.sqrt(mse_loss) + der_1st_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(sir.beta, 7.)
            torch.nn.utils.clip_grad_norm_(sir.gamma, 7.)
            torch.nn.utils.clip_grad_norm_(sir.delta, 7.)

            optimizer.step()
            if use_alpha:
                net_optimizer.step()

            if i % log_epoch_steps == 0:
                print(f"epoch {i} / {n_epochs}")
                mse_losses.append(mse_loss.detach().numpy())
                der_1st_losses.append(der_1st_loss.detach().numpy())
                der_2nd_losses.append(der_2nd_loss.detach().numpy())
                if summary is not None:
                    # add current plot of params
                    fig = sir.plot_params_over_time()
                    summary.add_figure("params_over_time", fig, close=True, global_step=i)
                    # add current fit
                    fig = sir.plot_sir_fit(w_hat, w_target)
                    summary.add_figure("sir fit", fig, close=True, global_step=i)

                    summary.add_scalar("losses/mse_loss", mse_loss, global_step=i)
                    summary.add_scalar("losses/tot_loss", total_loss, global_step=i)
                    summary.add_scalar("losses/der_1st_loss", der_1st_loss, global_step=i)
                    summary.add_scalar("losses/der_2nd_loss", der_2nd_loss, global_step=i)
                    summary.add_scalar("losses/beta_grad_norm", sir.beta.grad.detach().clone().norm(), global_step=i)
                    summary.add_scalar("losses/gamma_grad_norm", sir.gamma.grad.detach().clone().norm(), global_step=i)
                    summary.add_scalar("losses/delta_grad_norm", sir.delta.grad.detach().clone().norm(), global_step=i)

                print("Train Loss at step %d: %.7f" % (i, mse_loss))
                print("beta: " + str(sir.beta))
                print("gamma: " + str(sir.gamma))
                print("delta: " + str(sir.delta))
                time_step = time.time() - time_start
                time_start = time.time()
                print("Average time for epoch: {}".format(time_step / log_epoch_steps))

            if i % validation_epoch_steps == 0:
                with torch.no_grad():
                    val_w_hat, val_y_hat, _ = sir.inference(val_time_grid)
                    val_w_hat, val_y_hat = val_w_hat[val_hat_slice], val_y_hat[val_hat_slice]
                    val_mse_loss, val_w_loss, val_y_loss, val_total_loss = sir.loss(
                        val_w_hat, val_w_target, val_y_hat, val_y_target
                    )
                    print("Validation Loss at step %d: %.7f" % (i, val_mse_loss))
                    summary.add_scalar("losses/validation_mse_loss", val_mse_loss, global_step=i)
                if val_mse_loss + thresh < best:
                    # maintains the best solution found so far
                    best = val_mse_loss
                    best_beta = sir.beta.clone()
                    best_gamma = sir.gamma.clone()
                    best_delta = sir.delta.clone()
                    best_epoch = i
                    patience = 0
                elif patience < max_no_improve:
                    patience += 1
                elif n_lr_updts < max_n_lr_updts:
                    # when patience is over reduce learning rate by 2
                    print("Reducing learning rate at step: %d" % i)
                    lr_frac = 2.0
                    lr_b, lr_g, lr_d, lr_a = lr_b / lr_frac, lr_g / lr_frac, lr_d / lr_frac, lr_a / lr_frac
                    # optimizer.etas = [lr_b, lr_g, lr_d, lr_a, lr_a, lr_a, lr_a]
                    optimizer.etas = [lr_b, lr_g, lr_d]
                    n_lr_updts += 1
                    patience = 0
                else:
                    # after too many reductions early stops
                    print("Early stop at step: %d" % i)
                    break

        print("-" * 20)
        print("Best: " + str(best))
        print(sir.beta)
        print(sir.gamma)
        print(sir.delta)
        print(best_beta)
        print(best_gamma)
        print(best_delta)
        print("\n")

        sir.set_params(best_beta, best_gamma, best_delta)
        return sir, mse_losses, der_1st_losses, der_2nd_losses, best_epoch
