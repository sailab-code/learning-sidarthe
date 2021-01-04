import torch

from typing import Dict


from .sidarthe import Sidarthe


class SidartheExtended(Sidarthe):
    """
    Extended version of SIDARTHE Compartmental Model:
        Giordano, Giulia, et al.
        "Modelling the COVID-19 epidemic and implementation of population-wide interventions in Italy."
        Nature Medicine (2020): 1-6.

    As proposed in:
    Andrea Zugarini, Enrico Meloni et al.
    "An Optimal Control Approach to Learning in SIDARTHE Epidemic model."
    arXiv preprint arXiv:2010.14878 (2020).

    Two additional transitions are added: phi and chi, from (A,E) and (R,E) respectively.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    @property
    def param_groups(self) -> Dict:
        base_groups = super().param_groups
        base_groups['death_rates'] = ('tau', 'phi', 'chi')
        return base_groups

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


    def differential_equations(self, t, x):
        """
        Returns the right-hand side of extended SIDARTHE model
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

        p = {key: self.get_param_at_t(key, t) for key in self.params.keys()}

        S = x[0]
        I = x[1]
        D = x[2]
        A = x[3]
        R = x[4]
        T = x[5]

        S_dot = -S * (p['alpha'] * I + p['beta'] * D + p['gamma'] * A + p['delta'] * R)
        I_dot = -S_dot - (p['epsilon'] + p['zeta'] + p['lambda']) * I
        D_dot = p['epsilon'] * I - (p['eta'] + p['rho']) * D
        A_dot = p['zeta'] * I - (p['theta'] + p['mu'] + p['kappa'] + p['phi']) * A
        R_dot = p['eta'] * D + p['theta'] * A - (p['nu'] + p['xi'] + p['chi']) * R
        T_dot = p['mu'] * A + p['nu'] * R - (p['sigma'] + p['tau']) * T
        E_dot = p['phi'] * A + p['chi'] * R + p['tau'] * T
        H_det_dot = p['rho'] * D + p['xi'] * R + p['sigma'] * T


        return torch.cat((
            S_dot,
            I_dot,
            D_dot,
            A_dot,
            R_dot,
            T_dot,
            E_dot,
            H_det_dot
        ), dim=0)
