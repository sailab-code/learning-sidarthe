import numpy as np
import scipy.integrate as spi


class SirEq:
    def __init__(self, beta, gamma, delta, population, init_cond):
        self.beta = np.array(beta)
        self.gamma = np.array(gamma)
        self.delta = np.array(delta)
        self.population = population
        self.init_cond = init_cond

        self.b_reg = 1e7
        self.c_reg = 1e7
        self.d_reg = 1e7
        self.bc_reg = 1e7

    def dynamic_bc_diff_eqs(self, INP, t):
        """SIR Model with dynamic beta and gamma"""
        Y= np.zeros((3))
        V = INP
        t = int(t)
        beta = self.beta[t]/self.population if t < len(self.beta) else self.beta[-1]/self.population
        gamma = self.gamma[t] if t < len(self.gamma) else self.gamma[-1]
        Y[0] = - beta * V[0] * V[1]
        Y[1] = beta * V[0] * V[1] - gamma * V[1]
        Y[2] = gamma * V[1]
        return Y   # For odeint

    def diff_eqs(self, INP, t):
        """SIR Model"""
        Y = np.zeros((3))
        V = INP
        Y[0] = - self.beta/self.population * V[0] * V[1]
        Y[1] = self.beta/self.population * V[0] * V[1] - self.gamma * V[1]
        Y[2] = self.gamma * V[1]
        return Y   # For odeint

    def loss(self, x, y, diff_eqs):
        RES = spi.odeint(diff_eqs, self.init_cond, x)
        z = RES[:, 2]
        w_hat = self.delta * z
        # w_hat = w_hat[-1]
        # y = y[-1]
        mse_loss = np.sqrt(2 * np.mean(0.5 * (w_hat - y) * (w_hat - y)))
        b = np.abs(self.beta)
        c = np.abs(self.gamma)
        d = np.abs(self.delta)
        tot_loss = mse_loss + (np.greater_equal(b, 1.0) * b * self.b_reg) + \
                   (np.greater_equal(c, 1.0) * c * self.c_reg) + (np.greater_equal(d, 1.0) * d * self.d_reg) + \
                   (np.less_equal(self.beta, 0.0) * b * self.b_reg) + (np.less_equal(self.gamma, 0.0) * c * self.c_reg) + \
                   (np.less_equal(d, 0.0) * d * self.d_reg)

        return mse_loss, tot_loss, RES

    def inference(self, x, diff_eqs):
        RES = spi.odeint(diff_eqs, self.init_cond, x)
        z = RES[:, 2]

        delta = self.delta
        if not isinstance(delta, float) and len(delta) < len(z):
            delta = np.concatenate((delta, np.array([delta[-1]]*(len(z)-len(delta)))), axis=0)

        w_hat = delta * z

        return RES, w_hat

    def estimate_gradient(f, x, y, diff_eqs, h=5e-4):
        # _, f_0, _ = f.loss(x, y, diff_eqs)  # compute obj function
        old_beta = f.beta
        old_gamma = f.gamma
        old_delta = f.delta

        # df/d_beta
        f.beta = f.beta + h
        _, f_bh, _ = f.loss(x, y, diff_eqs)  # f(beta + h)
        f.beta = f.beta - h  # d_beta
        _, f_b_h, _ = f.loss(x, y, diff_eqs)  # f(beta - h)
        df_beta = (f_bh - f_b_h) / 2*h
        f.beta = old_beta

        # df/d_gamma
        f.gamma = f.gamma + h
        _, f_gh, _ = f.loss(x, y, diff_eqs)
        f.gamma = f.gamma - h
        _, f_g_h, _ = f.loss(x, y, diff_eqs)
        df_gamma = (f_gh - f_g_h) / 2*h
        f.gamma = old_gamma

        # df/d_delta
        f.delta = f.delta + h
        _, f_dh, _ = f.loss(x, y, diff_eqs)
        f.delta = f.delta - h
        _, f_d_h, _ = f.loss(x, y, diff_eqs)
        df_delta = (f_dh - f_d_h) / 2*h
        f.delta = old_delta

        return df_beta, df_gamma, df_delta

    def gradient_descent(self, x, y, diff_eqs, lr_b=1e-3, lr_g=1e-3, lr_d=1e-3):
        d_b, d_g, d_d = self.estimate_gradient(x, y, diff_eqs)
        self.beta -= lr_b * d_b
        self.gamma -= lr_g * d_g
        self.delta -= lr_d * d_d

        # eps = 1e-8
        #
        # self.beta = self.beta if 0.0 <= self.beta else eps
        # self.beta = self.beta if self.beta <= 1.0 else 1.0 - eps
        #
        # self.gamma = self.gamma if 0.0 <= self.gamma else eps
        # self.gamma = self.gamma if self.gamma <= 1.0 else 1.0 - eps
        #
        # self.delta = self.delta if 0.0 <= self.delta else eps
        # self.delta = self.delta if self.delta <= 1.0 else 1.0 - eps

    def updt_params(self, beta, gamma, delta):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    @staticmethod
    def train(target, y_0, z_0, params):
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
        sir = SirEq(beta, gamma, delta, population, INPUT)
        diff_eqs = sir.diff_eqs if params["eq_mode"] == "static" else sir.dynamic_bc_diff_eqs
        W = target[t_start:t_end]
        # W = target[:t_end]
        t_range = np.arange(t_start, t_end, t_inc)
        # t_range = np.arange(0, t_end, t_inc)
        best = 1e12
        thresh = 1e-5
        patience, n_lr_updts, max_no_improve, max_n_lr_updts = 0, 0, 25, 20
        best_beta, best_gamma, best_delta = sir.beta, sir.gamma, sir.delta
        for i in range(params["n_epochs"]):
            # loss, _, _ = sir.loss(t_range, W, t_start, t_end, diff_eqs)
            loss, _, _ = sir.loss(t_range, W, diff_eqs)
            # sir.gradient_descent(t_range, W, t_start, t_end, diff_eqs, lr_b, lr_g, lr_d)
            sir.gradient_descent(t_range, W, diff_eqs, lr_b, lr_g, lr_d)

            if i % 500 == 0:
                print("Loss at step %d: %.7f" % (i, loss))
                print("beta: " + str(sir.beta))
                print("gamma: " + str(sir.gamma))
                print("delta: " + str(sir.delta))
                print(Z0)
                print(W[-1])

            if loss < best + thresh:
                best = loss
                best_beta = sir.beta
                best_gamma = sir.gamma
                best_delta = sir.delta
                patience = 0
            elif patience < max_no_improve:
                patience += 1
            elif n_lr_updts < max_n_lr_updts:
                lr_b, lr_g, lr_d = lr_b/2, lr_g/2, lr_d/2
                n_lr_updts += 1
                patience = 0
            else:
                print("Early stop at step: %d" % i)
                break

        print("Best: " + str(best))
        print(best_beta)
        print(best_gamma)
        print(best_delta)
        print("\n")
        sir.updt_params(best_beta, best_gamma, best_delta)
        _, _, res = sir.loss(t_range, W, diff_eqs)

        return best, sir.beta, sir.gamma, sir.delta, sir, res
