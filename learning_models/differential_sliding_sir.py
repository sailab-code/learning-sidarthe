import numpy as np
import scipy.integrate as spi


class SirEq:
    def __init__(self, beta, gamma, delta, population, init_cond, mode="static", learning_setup="all_window"):
        """
        Sir model building
        :param beta: initial beta params
        :param gamma:
        :param delta:
        :param population: int. population size
        :param init_cond:
        :param mode: {"static"| "dynamic" | "joint_dynamic"}. If static, we assume
        to learn only one value for [beta,gamma,delta], therefore beta, gamma and delta are
        supposed to be lists of lenght 1. The same value over time is then used for differential equations.
         When mod is dynamic then we assume that beta,gamma and delta varies over time, hence each one is a list
         of size N >= 1.
         In case of joint_dynamic, the model uses variable parameters over time as in dynamic, but it can also
         learn as well to adjust jointly all the beta(t),gamma(t),delta(t) for each t.
        :param learning_setup: {all_window, last_only}: all_window setup tries to
        fit all the values of W, while in last_only the loss is computed only with the last value of the window.
        """

        self.beta = np.array(beta)
        print(beta)
        self.gamma = np.array(gamma)
        self.delta = np.array(delta)
        self.population = population
        self.init_cond = init_cond
        self.mode = mode
        self.learning_setup = learning_setup

        self.b_reg = 1e7
        self.c_reg = 1e7
        self.d_reg = 1e7
        self.bc_reg = 1e7
        self.ed_lambda = 0.7

        self.derivate_reg = 1e6
        self.second_derivate_reg = 0

    def dynamic_bc_diff_eqs(self, INP, t):
        """SIR Model with dynamic beta and gamma"""
        Y = np.zeros((3))
        V = INP
        t = int(t)
        beta = self.beta[t] / self.population if t < len(self.beta) else self.beta[-1] / self.population
        gamma = self.gamma[t] if t < len(self.gamma) else self.gamma[-1]
        Y[0] = - beta * V[0] * V[1]
        Y[1] = beta * V[0] * V[1] - gamma * V[1]
        Y[2] = gamma * V[1]
        return Y  # For odeint

    def diff_eqs(self, INP, t):
        """SIR Model"""
        Y = np.zeros((3))
        V = INP
        Y[0] = - self.beta / self.population * V[0] * V[1]
        Y[1] = self.beta / self.population * V[0] * V[1] - self.gamma * V[1]
        Y[2] = self.gamma * V[1]
        return Y  # For odeint

    def loss(self, x, y, diff_eqs):
        RES = spi.odeint(diff_eqs, self.init_cond, x)
        z = RES[:, 2]

        delta = self.delta
        if len(delta) < len(z):
            delta = np.concatenate((delta, np.array([delta[-1]] * (len(z) - len(delta)))), axis=0)

        w_hat = delta * z

        if self.learning_setup == "last_only":  # or self.learning_setup == "last_only_static":
            # just compute the loss on the last element of the window
            w_hat = w_hat[-1]
            y = y[-1]

        def parameter_derivate(parameter):
            return np.power((parameter[1] - parameter[0]), 2) + np.power(parameter[-1] - parameter[-2], 2)

        def loss_gte_one(parameter):
            return np.greater_equal(parameter, 1.0) * np.abs(parameter)

        def loss_lte_zero(parameter):
            return np.less_equal(parameter, 0.0) * np.abs(parameter)

        # mse_loss = np.sqrt(2 * np.mean(0.5 * (w_hat - y) * (w_hat - y)))  # MSE
        mse_loss = np.sqrt(2 * np.mean(0.5 * (w_hat - y) * (w_hat - y)))  # weighted MSE

        # compute losses due to derivate not close to zero near the window limits
        derivate_beta = parameter_derivate(self.beta)
        derivate_gamma = parameter_derivate(self.gamma)
        derivate_delta = parameter_derivate(self.delta)

        # second_derivates_beta = np.sum([self.second_order_derivative_regularization(self.beta, t) for t in range(1, len(self.beta)-1)])
        # second_derivates_gamma = np.sum([self.second_order_derivative_regularization(self.gamma, t) for t in range(1, len(self.gamma)-1)])
        # second_derivates_delta = np.sum([self.second_order_derivative_regularization(self.delta, t) for t in range(1, len(self.delta)-1)])
        # print(second_derivates_beta)
        # REGULARIZATION TO PREVENT b,c,d to go out of bounds
        # b = np.abs(self.beta)
        # c = np.abs(self.gamma)
        # +d = np.abs(self.delta)

        tot_loss = mse_loss + \
                   self.b_reg * (loss_gte_one(self.beta) + loss_lte_zero(self.beta)) + \
                   self.c_reg * (loss_gte_one(self.gamma) + loss_lte_zero(self.gamma)) + \
                   self.d_reg * (loss_gte_one(self.delta) + loss_lte_zero(self.delta)) + \
                   self.derivate_reg * (derivate_beta + derivate_gamma + derivate_delta)
                   # self.second_derivate_reg * (second_derivates_beta + second_derivates_gamma + second_derivates_delta)

        """
        tot_loss = mse_loss + (np.greater_equal(b, 1.0) * b * self.b_reg) + \
                   (np.greater_equal(c, 1.0) * c * self.c_reg) + (np.greater_equal(d, 1.0) * d * self.d_reg) + \
                   (np.less_equal(self.beta, 0.0) * b * self.b_reg) + (np.less_equal(self.gamma, 0.0) * c * self.c_reg) + \
                   (np.less_equal(d, 0.0) * d * self.d_reg) + \
                   self.derivate_reg * (derivate_beta + derivate_gamma + derivate_delta)
        """

        return mse_loss, tot_loss.mean(), RES

    def inference(self, x, diff_eqs):
        RES = spi.odeint(diff_eqs, self.init_cond, x)
        z = RES[:, 2]

        delta = self.delta
        if not isinstance(delta, float) and len(delta) < len(z):
            delta = np.concatenate((delta, np.array([delta[-1]] * (len(z) - len(delta)))), axis=0)

        w_hat = delta * z

        return RES, w_hat

    def estimate_gradient(f, x, y, diff_eqs, h=5e-4, t=-1):
        """
        Estimate gradient of beta, gamma and delta wrt the loss.
        :param x: input
        :param y:
        :param diff_eqs:
        :param h:
        :param t:
        :return:
        """
        # _, f_0, _ = f.loss(x, y, diff_eqs)  # compute obj function
        old_beta = f.beta[t]
        old_gamma = f.gamma[t]
        old_delta = f.delta[t]

        # df/d_beta
        f.beta[t] = f.beta[t] + h
        _, f_bh, _ = f.loss(x, y, diff_eqs)  # f(beta + h)
        f.beta[t] = f.beta[t] - 2 * h  # d_beta
        _, f_b_h, _ = f.loss(x, y, diff_eqs)  # f(beta - h)
        df_beta = (f_bh - f_b_h) / 2 * h  # f(b + h,g,d) - f(b - h,g,d) / 2h
        f.beta[t] = old_beta

        # df/d_gamma
        f.gamma[t] = f.gamma[t] + h
        _, f_gh, _ = f.loss(x, y, diff_eqs)
        f.gamma[t] = f.gamma[t] - 2 * h
        _, f_g_h, _ = f.loss(x, y, diff_eqs)
        df_gamma = (f_gh - f_g_h) / 2 * h  # f(b,g+h,d) - f(b,g+h,d) / 2h
        f.gamma[t] = old_gamma

        # df/d_delta
        f.delta[t] = f.delta[t] + h
        _, f_dh, _ = f.loss(x, y, diff_eqs)
        f.delta[t] = f.delta[t] - 2 * h
        _, f_d_h, _ = f.loss(x, y, diff_eqs)
        df_delta = (f_dh - f_d_h) / 2 * h  # f(b,g,d+h) - f(b,g,d-h) / 2h
        f.delta[t] = old_delta

        return df_beta, df_gamma, df_delta

    @staticmethod
    def second_order_derivative_regularization(parameter, t, h=1):
        return np.power((parameter[t+h] - 2*parameter[t] + parameter[t-h])/h**2,2)  # f(x+h) - 2*f(x) + f(x-h)/h^2

    def gradient_descent(self, x, y, diff_eqs, eta_b0=1e-3, eta_g0=1e-3, eta_d0=1e-3):

        if self.mode == "joint_dynamic":
            # updates all the betas, gammas and deltas at the same time
            d_b, d_g, d_d = [], [], []
            for t in range(len(self.beta)):
                d_b_t, d_g_t, d_d_t = self.estimate_gradient(x, y, diff_eqs, t=t)
                d_b.append(d_b_t)
                d_g.append(d_g_t)
                d_d.append(d_d_t)

            # print(d_b)
            # for t in range(len(self.beta)):
            #     self.beta[t] -= eta_b0 * d_b[t]
            #     self.gamma[t] -= eta_g0 * d_g[t]
            #     self.delta[t] -= eta_d0 * d_d[t]

            # mu = 0.0
            m_bt, m_gt, m_dt = 0.0, 0.0, 0.0
            a, b = 3.0, 0.05
            alpha = 1 / 7
            for t in range(len(self.beta)):
                mu = 1.0 / (1.0 + np.exp(-alpha * t))
                eta_decay = (a / (a - b * t))
                # eta_decay = 1 - mu
                eta_b = eta_b0 * eta_decay
                m_bt = -eta_b * d_b[t] + mu * m_bt

                eta_g = eta_g0 * eta_decay
                m_gt = -eta_g * d_g[t] + mu * m_gt

                eta_d = eta_d0 * eta_decay
                m_dt = -eta_d * d_d[t] + mu * m_dt

                self.beta[t] += m_bt
                self.gamma[t] += m_gt
                self.delta[t] += m_dt

                # self.beta[t] -= lr_b * d_b[t]
                # self.gamma[t] -= lr_g * d_g[t]
                # self.delta[t] -= lr_d * d_d[t]

        elif self.mode == "joint_dynamic_decay":
            # updates all the betas, gammas and deltas at the same time
            d_b, d_g, d_d = [], [], []
            for t in range(len(self.beta)):
                d_b_t, d_g_t, d_d_t = self.estimate_gradient(x, y, diff_eqs, t=t)
                d_b.append(d_b_t)
                d_g.append(d_g_t)
                d_d.append(d_d_t)

            for t in range(len(self.beta)):
                ti = len(self.beta) - 1 - t
                lr_b_d = np.exp(-self.ed_lambda * ti) * eta_b0
                lr_g_d = np.exp(-self.ed_lambda * ti) * eta_g0
                lr_d_d = np.exp(-self.ed_lambda * ti) * eta_d0
                self.beta[t] -= lr_b_d * d_b[t]
                self.gamma[t] -= lr_g_d * d_g[t]
                self.delta[t] -= lr_d_d * d_d[t]
        else:
            # updates only the last beta, gamma and delta
            t = -1
            d_b, d_g, d_d = self.estimate_gradient(x, y, diff_eqs, t=t)
            self.beta[t] -= eta_b0 * d_b
            self.gamma[t] -= eta_g0 * d_g
            self.delta[t] -= eta_d0 * d_d

    def updt_params(self, beta, gamma, delta):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    @staticmethod
    def train(target, y_0, z_0, params):
        """
        Static method to initialize a sir model with
         contitions and params specified, and thentrain it.
        :param target: a list of values to fit
        :param y_0: int initial infected population size
        :param z_0:  int initial population of recovered
        :param params:
        :return:
        """
        beta = params["beta"]
        gamma = params["gamma"]
        delta = params["delta"]
        population = params["population"]
        t_start = params["t_start"]
        t_end = params["t_end"]
        t_inc = 1

        lr_b, lr_g, lr_d = params["lr_b"], params["lr_g"], params["lr_d"]

        epsilon = y_0 / population  # 4.427e-4  # 1.66e-5  # set the seed of infection
        epsilon_z = z_0 / population
        # epsilon = y_0[0] / population  # 4.427e-4  # 1.66e-5  # set the seed of infection
        S0 = 1 - (epsilon + epsilon_z)
        I0 = epsilon
        Z0 = epsilon_z
        S0 = S0 * population
        I0 = I0 * population
        Z0 = Z0 * population
        INPUT = (S0, I0, Z0)  # initialization of SIR parameters (Suscettible, Infected, Recovered)
        sir = SirEq(beta, gamma, delta, population, INPUT, params["eq_mode"], learning_setup=params["learning_setup"])
        diff_eqs = sir.diff_eqs if params["eq_mode"] == "static" else sir.dynamic_bc_diff_eqs
        W = target[t_start:t_end]
        # W = target[:t_end]
        t_range = np.arange(t_start, t_end, t_inc)
        # t_range = np.arange(0, t_end, t_inc)

        # early stopping stuff
        best = 1e12
        thresh = 2e-1
        patience, n_lr_updts, max_no_improve, max_n_lr_updts = 0, 0, 3, 3

        losses = []
        der_1st_losses = []
        der_2nd_losses = []
        best_beta, best_gamma, best_delta = sir.beta, sir.gamma, sir.delta
        for i in range(params["n_epochs"]):
            # loss, _, _ = sir.loss(t_range, W, t_start, t_end, diff_eqs)
            loss, _, der_1st_loss, der_2nd_loss  = sir.loss(t_range, W, diff_eqs)
            # sir.gradient_descent(t_range, W, t_start, t_end, diff_eqs, lr_b, lr_g, lr_d)
            sir.gradient_descent(t_range, W, diff_eqs, lr_b, lr_g, lr_d)

            if i % 50 == 0:
                losses.append(loss)
                der_1st_losses.append(der_1st_loss)
                der_2nd_losses.append(der_2nd_loss)
                print("Loss at step %d: %.7f" % (i, loss))
                print("beta: " + str(sir.beta))
                print("gamma: " + str(sir.gamma))
                print("delta: " + str(sir.delta))
                print(Z0)
                print(W[-1])

                if loss + thresh < best:
                    # maintains the best solution found so far
                    best = loss
                    best_beta = sir.beta
                    best_gamma = sir.gamma
                    best_delta = sir.delta
                    patience = 0
                elif patience < max_no_improve:
                    patience += 1
                elif n_lr_updts < max_n_lr_updts:
                    # when patience is over reduce learning rate by 2
                    lr_b, lr_g, lr_d = lr_b / 2, lr_g / 2, lr_d / 2
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
        sir.updt_params(best_beta, best_gamma, best_delta)  # assign the best params to the model
        _, _, res = sir.loss(t_range, W, diff_eqs)

        return best, sir.beta[-1], sir.gamma[-1], sir.delta[-1], sir, res, losses, der_1st_losses, der_2nd_losses
