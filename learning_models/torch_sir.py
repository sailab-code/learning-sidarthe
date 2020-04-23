import time

import numpy
import torch
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch_euler import euler
from torchdiffeq import odeint


class SirOptimizer(Optimizer):
    def __init__(self, params, etas, alpha, a, b):
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
        defaults = dict()

        super(SirOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None,
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            times = torch.arange(group["params"][0].shape[0], dtype=torch.float32)

            # mu = torch.sigmoid(1 / self.alpha * (times / max(times) - 0.5))
            mu = torch.sigmoid(self.alpha * times)
            # print(mu)
            eta_mod = self.a / (self.a + self.b * times)
            # eta_mod = 1 - mu
            etas = torch.tensor(self.etas)

            etas = etas.unsqueeze(1) * eta_mod.unsqueeze(0)  # modded learning rates
            for idx, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                # print(p.grad)

                d_p = parameter.grad.data

                update = [-etas[idx][0] * d_p[0]]
                for t in range(1, d_p.size(0)):
                    momentum_term = -etas[idx][t] * d_p[t] + mu[t] * update[t-1]
                    update.append(momentum_term)


                # print("UPDATE: {}".format(update))
                #parameter.data.add_(-self.etas[idx] * d_p)
                parameter.data.add_(torch.tensor(update))


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
        self.der_1st_reg = kwargs.get("derivative_reg", 1000)
        self.der_2nd_reg = kwargs.get("der_2nd_reg", 1000)

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
        t = T.long()

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

    def loss(self, w_hat, w_target):
        if isinstance(w_target, numpy.ndarray):
            w_target = torch.tensor(w_target, dtype=w_hat.dtype)

        def first_derivative_central(f_x_plus_h, f_x_minus_h, h):
            return torch.pow((f_x_plus_h - f_x_minus_h) / (2 * h), 2)

        def first_derivative_forward(f_x_plus_h, f_x, h):
            return torch.pow((f_x_plus_h - f_x) / h, 2)

        def first_derivative_backward(f_x, f_x_minus_h, h):
            return torch.pow((f_x - f_x_minus_h) / h, 2)

        h = 1  # todo: retrieve sampling

        def loss_derivative(parameter: torch.Tensor):
            forward = first_derivative_forward(parameter[1], parameter[0], h).unsqueeze(0)
            central = first_derivative_central(parameter[2:], parameter[:-2], h)
            backward = first_derivative_backward(parameter[-1], parameter[-2], h).unsqueeze(0)

            return central
            # return torch.cat((forward, central, backward), dim=0)

        def loss_gte_one(parameter: torch.Tensor):
            return torch.where(parameter.ge(1.), torch.ones(1), torch.zeros(1)) * parameter.abs()

        def loss_lte_zero(parameter: torch.Tensor):
            return torch.where(parameter.le(0.), torch.ones(1), torch.zeros(1)) * parameter.abs()

        def second_derivative_central(f_x_plus_h, f_x, f_x_minus_h):
            return f_x_plus_h - 2 * f_x + f_x_minus_h

        def second_derivative_forward(f_x_plus_2h, f_x_plus_h, f_x):
            return f_x_plus_2h - 2 * f_x_plus_h + f_x

        def second_derivative_backward(f_x, f_x_minus_h, f_x_minus_2h):
            return f_x - 2 * f_x_minus_h + f_x_minus_2h

        def loss_second_derivative(parameter: torch.Tensor):
            forward = second_derivative_forward(parameter[2], parameter[1], parameter[0]).unsqueeze(0)
            central = second_derivative_central(parameter[2:], parameter[1:-1], parameter[:-2])
            backward = second_derivative_backward(parameter[-1], parameter[-2], parameter[-3]).unsqueeze(0)
            """
                        loss = torch.tensor([0.])
                        for t, value in enumerate(parameter):
                            if t == 0:
                                loss = loss + second_derivative_forward(parameter[t + 2], parameter[t + 1], parameter[t])
                            elif t < parameter.shape[0] - 1:
                                loss = loss + second_derivative_central(parameter[t + 1], parameter[t], parameter[t - 1])
                            else:
                                loss = loss + second_derivative_backward(parameter[t], parameter[t - 1], parameter[t - 2])

                        return loss
            """
            return central
            # return torch.cat((forward, central, backward), dim=0)
        mse_loss = torch.sqrt(2 * torch.mean(0.5 * torch.pow((w_hat - w_target), 2)))

        # compute losses due to derivative not close to zero near the window limits
        loss_1st_derivative_beta = loss_derivative(self.beta)
        loss_1st_derivative_gamma = loss_derivative(self.gamma)
        loss_1st_derivative_delta = loss_derivative(self.delta)
        loss_1st_derivative_total = self.der_1st_reg * torch.mean(
            loss_1st_derivative_beta + loss_1st_derivative_gamma + loss_1st_derivative_delta)

        # compute losses due to second derivative
        loss_2nd_derivative_beta = loss_second_derivative(self.beta)
        loss_2nd_derivative_gamma = loss_second_derivative(self.gamma)
        loss_2nd_derivative_delta = loss_second_derivative(self.delta)
        loss_2nd_derivative_total = self.der_2nd_reg * torch.mean(
            loss_2nd_derivative_beta + loss_2nd_derivative_gamma + loss_2nd_derivative_delta)

        # REGULARIZATION TO PREVENT b,c,d from going out of bounds
        loss_reg_beta = self.b_reg * (loss_gte_one(self.beta) + loss_lte_zero(self.beta))
        loss_reg_gamma = self.c_reg * (loss_gte_one(self.gamma) + loss_lte_zero(self.gamma))
        loss_reg_delta = self.d_reg * (loss_gte_one(self.delta) + loss_lte_zero(self.delta))

        # compute total loss
        total_loss = mse_loss + \
                     loss_reg_beta + loss_reg_gamma + loss_reg_delta

        total_loss = total_loss + loss_1st_derivative_total

        # total_loss = total_loss + loss_2nd_derivative_total

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
        n_epochs = params.get("n_epochs", 2000)
        t_inc = params.get("t_inc", 1)
        lr_b, lr_g, lr_d = params["lr_b"], params["lr_g"], params["lr_d"]

        w_target = torch.tensor(target[t_start:t_end], dtype=torch.float32)
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
                    b_reg=b_reg, c_reg=c_reg, d_reg=d_reg)

        # early stopping stuff
        best = 1e12
        thresh = 1e-5
        patience, n_lr_updts, max_no_improve, max_n_lr_updts = 0, 0, 75, 20
        best_beta, best_gamma, best_delta = sir.beta, sir.gamma, sir.delta

        # todo: implement custom optimizer
        # optimizer = SGD(sir.params(), lr=1e-9)
        optimizer = SirOptimizer(sir.params(), [lr_b, lr_g, lr_d], alpha=1 / 10, a=1.0, b=0.05)

        time_start = time.time()
        for i in range(n_epochs):
            w_hat, _ = sir.inference(time_grid)
            w_hat = w_hat[slice(t_start,int(t_end/t_inc),int(1/t_inc))]
            optimizer.zero_grad()
            mse_loss, total_loss = sir.loss(w_hat, w_target)

            total_loss.backward()
            # mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(sir.params(), 10)
            # print(f"after: \n {sir.beta.grad} \n \n")
            optimizer.step()

            if i % 50 == 0:
                print("Loss at step %d: %.7f" % (i, mse_loss))
                print("beta: " + str(sir.beta.grad))
                print("gamma: " + str(sir.gamma.grad))
                print("delta: " + str(sir.delta.grad))
                time_step = time.time() - time_start
                time_start = time.time()
                print(f"Average time for epoch: {time_step / 50}")
                # print(Z0)
                # print(w_hat[-1])

            if mse_loss + thresh < best:
                # maintains the best solution found so far
                best = mse_loss
                best_beta = sir.beta
                best_gamma = sir.gamma
                best_delta = sir.delta
                patience = 0
            elif patience < max_no_improve:
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
                break

        print("Best: " + str(best))
        print(best_beta)
        print(best_gamma)
        print(best_delta)
        print("\n")

        return sir
