from typing import Dict
from collections import namedtuple

import torch

from learning_models.tied_sidarthe_extended import TiedSidartheExtended
from utils.visualization_utils import Curve, generic_plot

Param = namedtuple("Param", ["value", "length"])
EPS = 1e-5

class StepwiseTiedSidartheExtended(TiedSidartheExtended):
    dtype = torch.float64

    def __init__(self, parameters: Dict, population, init_cond, integrator, sample_time, **kwargs):
        super().__init__(parameters, population, init_cond, integrator, sample_time, **kwargs)
        self.model_name = kwargs.get("name", "stepwise_tied_sidarthe_extended")

    def set_initial_params(self, params):
        self._params = {key: torch.tensor([p.value for p in param], dtype=self.dtype, requires_grad=True) for key, param in params.items()}
        self._params_map = {key: [i for i, p in enumerate(param) for _ in range(p.length)] for key, param in params.items()}

    def get_param_at_t(self, param_name, _t):
        t = int(_t.long().numpy())
        param_map = self._params_map[param_name]
        actual_param_id = param_map[t] if 0 <= t < len(param_map) else -1

        return torch.relu(self.params[param_name][actual_param_id].unsqueeze(0)) + EPS

    def extend_param(self, param_name, length):
        ext_tensor = torch.zeros(length)
        for t in range(length):
            actual_param_id = self._params_map[param_name][t] if 0 <= t < len(self._params_map[param_name]) else -1
            ext_tensor[t].fill_(float(self.params[param_name][actual_param_id].detach().numpy()))

        return torch.relu(ext_tensor) + EPS

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

        alpha = self.get_param_at_t("alpha", t) / self.population
        beta = self.get_param_at_t("beta", t) / self.population
        gamma = self.get_param_at_t("gamma", t) / self.population
        delta = self.get_param_at_t("delta", t) / self.population
        epsilon = self.get_param_at_t("epsilon", t)
        theta = self.get_param_at_t("theta", t)
        xi = self.get_param_at_t("xi", t)
        eta = self.get_param_at_t("eta", t)
        mu = self.get_param_at_t("mu", t)
        nu = self.get_param_at_t("nu", t)
        tau = self.get_param_at_t("tau", t)
        lambda_ = self.get_param_at_t("lambda", t)
        kappa = self.get_param_at_t("kappa", t)
        zeta = self.get_param_at_t("zeta", t)
        rho = self.get_param_at_t("rho", t)
        sigma = self.get_param_at_t("sigma", t)
        phi = self.get_param_at_t("phi", t)
        chi = self.get_param_at_t("chi", t)
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

        extended_params = {key: self.extend_param(key, time_grid.shape[0]) for key, _ in self.params.items()}

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

    def plot_params_over_time(self, n_days=None):
        param_plots = []

        if n_days is None:
            n_days = self.train_size

        for param_group, param_keys in self.param_groups.items():
            params_subdict = {param_key: self.params[param_key] for param_key in param_keys}
            for param_key, param in params_subdict.items():
                param = self.extend_param(param_key, n_days)
                pl_x = list(range(n_days))
                pl_title = f"{param_group}/$\\{param_key}$ over time"
                param_curve = Curve(pl_x, param.detach().numpy(), '-', f"$\\{param_key}$", color=None)
                curves = [param_curve]

                if self.references is not None:
                    if param_key in self.references:
                        ref_curve = Curve(pl_x, self.references[param_key][:n_days], "--", f"$\\{param_key}$ reference", color=None)
                        curves.append(ref_curve)
                plot = generic_plot(curves, pl_title, None, formatter=self.format_xtick)
                param_plots.append((plot, pl_title))
        return param_plots