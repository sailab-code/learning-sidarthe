import torch
from learning_models.sidarthe_extended import SidartheExtended

EPS = 0


class SpatioTemporalSidarthe(SidartheExtended):
    def __init__(self, parameters: Dict, population, init_cond, integrator, sample_time, **kwargs):
        super().__init__(parameters, population, init_cond, integrator, sample_time, **kwargs)
        self.model_name = kwargs.get("name", "spatio_temporal_sidarthe_extended")

        self.batch_size = kwargs.get("n_areas", None) # fixme name not appropriate


    def omega(self, t):
        if t >= 0:
            return torch.tensor([self.init_cond[:, :8]], dtype=self.dtype)  # 1 x s x p, p=8 number of states in sidarthe
        else:
            return torch.tensor([[[1.] + [0.] * 7]], dtype=self.dtype)

    @staticmethod
    def get_param_at_t(param, _t):
        """
        retrieves the value of a param at time t for all s.
        :param param: t x s tensor
        :param _t: int representing the current time step t
        :return: the value of the param at time t for all s.
        """
        _t = _t.long()
        if 0 <= _t < param.shape[0]:
            rectified_param = torch.relu(param[_t, :].unsqueeze(0))
        else:
            rectified_param = torch.relu(param[-1, :].unsqueeze(0).detach())

        return torch.where(rectified_param >= EPS, rectified_param, rectified_param + EPS)  # shape 1 x s (or 1 it is broadcasted) x

    def differential_equations(self, t, x):
        """
        Returns the right-hand side of SIDARTHE model
        :param t: time t at which right-hand side is computed
        :param x: state of model at time t forall the fitted areas.
            x[:,0] = S
            x[:,1] = I
            x[:,2] = D
            x[:,3] = A
            x[:,4] = R
            x[:,5] = T
            x[:,6] = H
            x[:,7] = E
        :return: right-hand side of SIDARTHE model, i.e. f(t,:,x(t)), the second dimension correspond to the areas to fit.
        """

        get_param_at_t = self.get_param_at_t

        # region parameters
        alpha = get_param_at_t(self.alpha, t) / self.population
        beta = get_param_at_t(self.beta, t) / self.population
        gamma = get_param_at_t(self.gamma, t) / self.population
        delta = get_param_at_t(self.delta, t) / self.population

        epsilon = get_param_at_t(self.epsilon, t)
        theta = get_param_at_t(self.theta, t)
        xi = get_param_at_t(self.xi, t)
        eta = get_param_at_t(self.eta, t)
        mu = get_param_at_t(self.mu, t)
        nu = get_param_at_t(self.nu, t)
        tau = get_param_at_t(self.tau, t)
        lambda_ = get_param_at_t(self.lambda_, t)
        kappa = get_param_at_t(self.kappa, t)
        zeta = get_param_at_t(self.zeta, t)
        rho = get_param_at_t(self.rho, t)
        sigma = get_param_at_t(self.sigma, t)
        phi = get_param_at_t(self.phi, t)
        chi = get_param_at_t(self.chi, t)

        S = x[:, 0]
        I = x[:, 1]
        D = x[:, 2]
        A = x[:, 3]
        R = x[:, 4]
        T = x[:, 5]
        E = x[:, 6]
        H_detected = x[:, 7]
        # H = x[7]

        # Differential Equations
        S_dot = -S * (alpha * I + beta * D + gamma * A + delta * R)
        I_dot = -S_dot - (epsilon + zeta + lambda_) * I
        D_dot = epsilon * I - (eta + rho) * D
        A_dot = zeta * I - (theta + mu + kappa + phi) * A
        R_dot = eta * D + theta * A - (nu + xi + chi) * R
        T_dot = mu * A + nu * R - (sigma + tau) * T
        E_dot = phi * A + chi * R + tau * T
        H_detected = rho * D + xi * R + sigma * T
        # H_dot = lambda_ * I + rho * D + kappa * A + zeta * R + sigma * T

        return torch.cat((
            S_dot,
            I_dot,
            D_dot,
            A_dot,
            R_dot,
            T_dot,
            E_dot,
            H_detected
            # H_dot,
        ), dim=0)

    def inference(self, time_grid) -> Dict:
        sol = self.integrate(time_grid)

        s = sol[:, :, 0]
        i = sol[:, :, 1]
        d = sol[:, :, 2]
        a = sol[:, :, 3]
        r = sol[:, :, 4]
        t = sol[:, :, 5]
        e = sol[:, :, 6]
        h_detected = sol[:, :, 7]
        h = self.population - (s + i + d + a + r + t + e)

        rt = self.get_rt(time_grid)

        return {
            "s": s,
            "i": i,
            "d": d,
            "a": a,
            "r": r,
            "t": t,
            "h": h,
            "e": e,
            "h_detected": h_detected,
            "r0": rt,
            "sol": sol
        }

    def extend_param(self, value, length):
        """

        :param value: a 2-d tensor, t|1 x s|1
        :param length: int temporal size to extend
        :return:
        """

        t_diff = self.train_size - value.shape[0]
        ext_t_tensor = torch.tensor([value[-1,:]] * t_diff)
        ext_t_tensor = torch.cat((value, ext_t_tensor), dim=0) # T x s

        s_diff = self.batch_size - value.shape[1]
        ext_s_tensor = torch.tensor([value[:, -1]] * s_diff)
        ext_tensor = torch.cat((ext_t_tensor, ext_s_tensor), dim=1)  # T x S

        rectified_param = torch.relu(ext_tensor)
        return torch.where(rectified_param >= EPS, rectified_param, rectified_param + EPS)

    def get_rt(self, time_grid):
        extended_params = {key: self.extend_param(value, time_grid.shape[0]) for key, value in self.params.items()}

        # compute R0
        c1 = extended_params['epsilon'] + extended_params['zeta'] + extended_params['lambda']
        c2 = extended_params['eta'] + extended_params['rho']
        c3 = extended_params['theta'] + extended_params['mu'] + extended_params['kappa'] + extended_params['phi']
        c4 = extended_params['nu'] + extended_params['xi'] + extended_params['chi']

        rt = extended_params['alpha'] + extended_params['beta'] * extended_params['epsilon'] / c2
        rt = rt + extended_params['gamma'] * extended_params['zeta'] / c3
        rt = rt + extended_params['delta'] * (extended_params['eta'] * extended_params['epsilon']) / (c2 * c4)
        rt = rt + extended_params['delta'] * extended_params['zeta'] * extended_params['theta'] / (c3 * c4)
        return rt / c1

