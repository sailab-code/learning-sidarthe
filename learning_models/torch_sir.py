import numpy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch_euler import euler


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

                # # print(p.grad)
                # d_p = parameter.grad.data
                # lr_t = -etas[idx] * d_p
                # mu_t = torch.cumprod(torch.flip(torch.cat((torch.ones(1), mu[:-1])), dims=[0]), dim=0)  # 1, mu[0], mu[0]*mu[1], ...
                # update = torch.cumsum(lr_t * mu_t, dim=0)

                d_p = parameter.grad.data
                update = [torch.tensor(-etas[idx][0] * d_p[0])]
                for t in range(1, d_p.size(0)):
                    momentum_term = -etas[idx][t] * d_p[t] + mu[t] * update[t-1]
                    update.append(momentum_term)



                # print("UPDATE: {}".format(update))
                # parameter.data.add_(-self.etas[idx] * d_p)
                parameter.data.add_(torch.tensor(update))


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
        self.der_1st_reg = kwargs.get("derivative_reg", 1e4)
        self.der_2nd_reg = kwargs.get("der_2nd_reg", 1e1)

        use_alpha = kwargs.get("use_alpha", False)
        input_size = kwargs.get("mlp_input", 4)
        hidden_size = kwargs.get("mlp_hidden", 3)
        self.init_alpha(use_alpha, input_size=input_size, hidden_size=hidden_size)

        if mode == "dynamic":
            self.diff_eqs = self.dynamic_diff_eqs
        else:
            self.diff_eqs = self.static_diff_eqs

    def get_policy_code(self, t):
        policy_code = torch.zeros(4)
        if 6 < t <= 13:
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
            return [1, 0, 0]

    def dynamic_diff_eqs(self, T, X, dt):
        X_t = X(T)
        t = T.long()

        policy_code = self.get_policy_code(t)

        if t < self.beta.shape[0]:
            beta = (self.beta[t] / self.population) * self.alpha(policy_code).squeeze()
            gamma = self.gamma[t]
        else:
            beta = (self.beta[-1] / self.population) * self.alpha(policy_code).squeeze()
            gamma = self.gamma[-1]

        return [
            - beta * X_t[0] * X_t[1],
            beta * X_t[0] * X_t[1] - gamma * X_t[1],
            gamma * X_t[1]
        ]

    def static_diff_eqs(self, T, X, dt):
        X_t = X(T)
        t = T.long()

        policy_code = self.get_policy_code(t)

        beta = self.alpha(policy_code) * self.beta / self.population
        gamma = self.gamma

        return [
            - beta * X_t[0] * X_t[1],
            beta * X_t[0] * X_t[1] - gamma * X_t[1],
            gamma * X_t[1]
        ]

    def init_alpha(self, use_alpha, input_size, hidden_size, output_size=1):
        """mlp"""

        if use_alpha:
            self.alpha = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size),
                                       nn.Linear(in_features=hidden_size, out_features=output_size),
                                       nn.Sigmoid())
        else:
            self.alpha = self._foo_alpha

    def _foo_alpha(self, policy):
        return torch.tensor([1.0])

    def loss(self, w_hat, w_target, y_hat=None, y_target=None):
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

        w_mse_loss = torch.sqrt(2 * torch.mean(0.5 * torch.pow((w_hat - w_target), 2)))

        if y_hat is not None and y_target is not None:
            y_mse_loss = torch.sqrt(2 * torch.mean(0.5 * torch.pow((y_hat - y_target), 2)))
        else:
            y_mse_loss = 0.0

        # compute losses due to derivative not close to zero near the window limits
        loss_1st_derivative_beta = loss_derivative(self.beta)
        loss_1st_derivative_gamma = loss_derivative(self.gamma)
        loss_1st_derivative_delta = 0.0  # loss_derivative(self.delta)
        loss_1st_derivative_total = self.der_1st_reg * torch.mean(loss_1st_derivative_beta + loss_1st_derivative_gamma + loss_1st_derivative_delta)

        # compute losses due to second derivative
        loss_2nd_derivative_beta = loss_second_derivative(self.beta)
        loss_2nd_derivative_gamma = loss_second_derivative(self.gamma)
        loss_2nd_derivative_delta = 0.0  # loss_second_derivative(self.delta)
        loss_2nd_derivative_total = self.der_2nd_reg * torch.mean(loss_2nd_derivative_beta + loss_2nd_derivative_gamma + loss_2nd_derivative_delta)

        # REGULARIZATION TO PREVENT b,c,d from going out of bounds
        loss_reg_beta = self.b_reg * (loss_gte_one(self.beta) + loss_lte_zero(self.beta))
        loss_reg_gamma = self.c_reg * (loss_gte_one(self.gamma) + loss_lte_zero(self.gamma))
        loss_reg_delta = self.d_reg * (loss_gte_one(self.delta) + loss_lte_zero(self.delta))

        # compute total loss
        y_target_scale = 0.0
        mse_loss = ((1.0 - y_target_scale) * w_mse_loss) + y_target_scale*y_mse_loss
        total_loss = mse_loss +\
                     loss_reg_beta + loss_reg_gamma + loss_reg_delta + \
                     loss_1st_derivative_total + loss_2nd_derivative_total

        return mse_loss, torch.mean(total_loss)

    def inference(self, time_grid):
        sol = euler(self.diff_eqs, self.omega, time_grid)
        z_hat = sol[:, 2]

        delta = self.delta
        len_diff = z_hat.shape[0] - delta.shape[0]
        if len_diff > 0:
            delta = torch.cat((delta, delta[-1].expand(len_diff)))

        w_hat = delta * z_hat

        return w_hat, sol[:, 1], sol

    def params(self):
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
        b_reg = params.get("b_reg", 1e7)
        c_reg = params.get("c_reg", 1e7)
        d_reg = params.get("d_reg", 1e8)
        n_epochs = params.get("n_epochs", 2000)
        derivative_reg = params.get("derivative_reg", 1e5)
        der_2nd_reg = params.get("der_2nd_reg", 1e5)
        use_alpha = params.get("use_alpha", True)

        t_inc = 1
        lr_b, lr_g, lr_d = params["lr_b"], params["lr_g"], params["lr_d"]

        time_grid = torch.arange(t_start, t_end+t_inc, t_inc)
        w_target = torch.tensor(w_target[t_start:t_end], dtype=torch.float32)
        y_target = torch.tensor(y_target[t_start:t_end], dtype=torch.float32)

        # init parameters
        epsilon = y_target[t_start].item() / population
        epsilon_z = w_target[t_start].item() / population
        S0 = 1 - (epsilon + epsilon_z)
        I0 = epsilon
        S0 = S0 * population
        I0 = I0 * population
        Z0 = epsilon_z

        init_cond = (S0, I0, Z0)  # initialization of SIR parameters (Suscettible, Infected, Recovered)
        sir = SirEq(beta, gamma, delta, population, init_cond,
                    b_reg=b_reg, c_reg=c_reg, d_reg=d_reg, derivative_reg=derivative_reg, der_2nd_reg=der_2nd_reg, use_alpha=use_alpha)

        # early stopping stuff
        best = 1e12
        thresh = 1e-6
        patience, n_lr_updts, max_no_improve, max_n_lr_updts = 0, 0, 25, 25
        best_beta, best_gamma, best_delta = sir.beta, sir.gamma, sir.delta

        # todo: implement custom optimizer
        # optimizer = SGD(sir.params(), lr=1e-9)
        optimizer = SirOptimizer(sir.params(), [lr_b, lr_g, lr_d], alpha=1 / 7, a=3.0, b=0.05)

        losses = []
        for i in range(n_epochs):
            w_hat, y_hat, _ = sir.inference(time_grid)
            optimizer.zero_grad()
            mse_loss, total_loss = sir.loss(w_hat, w_target, y_hat, y_target)

            total_loss.backward()
            # mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(sir.params(), 7.0)
            # print(f"after: \n {sir.beta.grad} \n \n")
            optimizer.step()

            if i % 50 == 0:
                losses.append(mse_loss.detach().numpy())
                print("Loss at step %d: %.7f" % (i, mse_loss))
                print("beta: " + str(sir.beta))
                print("gamma: " + str(sir.gamma))
                print("delta: " + str(sir.delta))
                # print("alpha: " + str([sir.alpha(sir.get_policy_code(t)).detach().numpy() for t in range(sir.beta.shape[0])]))
                # print(Z0)
                # print(w_hat[-1])

            if mse_loss + thresh < best:
                # maintains the best solution found so far
                best = mse_loss
                best_beta = sir.beta.clone()
                best_gamma = sir.gamma.clone()
                best_delta = sir.delta.clone()
                patience = 0
            elif patience < max_no_improve:
                patience += 1
            elif n_lr_updts < max_n_lr_updts:
                # when patience is over reduce learning rate by 2
                print("Reducing learning rate at step: %d" % i)
                lr_frac = 2.0
                lr_b, lr_g, lr_d = lr_b / lr_frac, lr_g / lr_frac, lr_d / lr_frac
                optimizer.etas = [lr_b, lr_g, lr_d]
                n_lr_updts += 1
                patience = 0
            else:
                # after too many reductions early stops
                print("Early stop at step: %d" % i)
                break

        print("Best: " + str(best))
        print(sir.beta)
        print(sir.gamma)
        print(sir.delta)
        print(best_beta)
        print(best_gamma)
        print(best_delta)
        print("\n")

        sir.set_params(best_beta, best_gamma, best_delta)

        return sir, losses
