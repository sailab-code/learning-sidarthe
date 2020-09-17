from typing import Dict

import torch

from learning_models.sidarthe import Sidarthe


class SidartheExtended(Sidarthe):
    dtype = torch.float64

    def __init__(self, parameters: Dict, population, init_cond, integrator, sample_time, **kwargs):
        super().__init__(parameters, population, init_cond, integrator, sample_time, **kwargs)
        self.model_name = kwargs.get("name", "sidarthe_extended")

    @property
    def params(self) -> Dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "theta": self.theta,
            "xi": self.xi,
            "eta": self.eta,
            "mu": self.mu,
            "nu": self.nu,
            "tau": self.tau,
            "lambda": self.lambda_,
            "kappa": self.kappa,
            "zeta": self.zeta,
            "rho": self.rho,
            "sigma": self.sigma,
            "phi": self.phi,
            "chi": self.chi
        }

    # region ModelParams

    @property
    def phi(self) -> torch.Tensor:
        return self._params["phi"]

    @property
    def chi(self) -> torch.Tensor:
        return self._params["chi"]

    # endregion CodeParams

    def differential_equations(self, t, x):
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

        def get_param_at_t(param, _t):
            _t = _t.long()
            if 0 <= _t < param.shape[0]:
                return param[_t].unsqueeze(0)
            else:
                return param[-1].unsqueeze(0)

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
        # endregion parameters

        S = x[0]
        I = x[1]
        D = x[2]
        A = x[3]
        R = x[4]
        T = x[5]
        E = x[6]
        H_detected = x[7]
        # H = x[7]

        # region equations

        S_dot = -S * (alpha * I + beta * D + gamma * A + delta * R)
        I_dot = -S_dot - (epsilon + zeta + lambda_) * I
        D_dot = epsilon * I - (eta + rho) * D
        A_dot = zeta * I - (theta + mu + kappa + phi) * A
        R_dot = eta * D + theta * A - (nu + xi + chi) * R
        T_dot = mu * A + nu * R - (sigma + tau) * T
        E_dot = phi * A + chi * R + tau * T
        H_detected = rho * D + xi * R + sigma * T
        # H_dot = lambda_ * I + rho * D + kappa * A + zeta * R + sigma * T

        # endregion equations

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
        s = sol[:, 0]
        i = sol[:, 1]
        d = sol[:, 2]
        a = sol[:, 3]
        r = sol[:, 4]
        t = sol[:, 5]
        e = sol[:, 6]
        h_detected = sol[:, 7]
        h = self.population - (s + i + d + a + r + t + e)

        extended_params = {key: self.extend_param(value, time_grid.shape[0]) for key, value in self.params.items()}

        # region compute R0
        c1 = extended_params['epsilon'] + extended_params['zeta'] + extended_params['lambda']
        c2 = extended_params['eta'] + extended_params['rho']
        c3 = extended_params['theta'] + extended_params['mu'] + extended_params['kappa'] + extended_params['phi']
        c4 = extended_params['nu'] + extended_params['xi'] + extended_params['chi']

        r0 = extended_params['alpha'] + extended_params['beta'] * extended_params['epsilon'] / c2
        r0 = r0 + extended_params['gamma'] * extended_params['zeta'] / c3
        r0 = r0 + extended_params['delta'] * (extended_params['eta'] * extended_params['epsilon']) / (c2 * c4)
        r0 = r0 + extended_params['delta'] * extended_params['zeta'] * extended_params['theta'] / (c3 * c4)
        r0 = r0 / c1
        # endregion

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
            "r0": r0,
            "sol": sol
        }
