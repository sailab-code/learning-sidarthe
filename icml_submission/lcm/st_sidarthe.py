from typing import Dict

import torch
from lcm.sidarthe_extended import SidartheExtended


class SpatioTemporalSidarthe(SidartheExtended):
    """
    Spatio-temporal version of SIDARTHE.
    This version generalizes the previous time-dependent formulation, therefore is also
    a generalization of its initial constant (or step-wise constant) definition.

    Features:
        1. Parallelization of ODE computation to S areas.
        2. Training of one model jointly on different areas.
        3. Definition SIDARTHE parameters as function of time and SPACE.
            E.g. alpha -> alpha(s,t)
            3.1 The case of time-dependent only parameters is still supported.
                E.g. alpha -> alpha(.,t)
            3.2 Constant parameters are still supported.
                E.g. alpha

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.model_name = kwargs.get("name", "spatio_temporal_sidarthe_extended")
        self.n_areas = kwargs.get("n_areas", None)

    def get_description(self):
        base_description = super().get_description()
        return {
            **base_description,
            "n_areas": self.n_areas
        }

    def get_param_at_t(self, param_key, _t):
        """
        Retrieves the value of a param at time t for all s.
        :param param: t x s tensor
        :param _t: int representing the current time step t
        :return: the value of the param at time t for all s.
        """

        param = self.params[param_key]
        _t = _t.long()
        # TODO check if possible to improve here
        if 0 <= _t[0] < param.shape[0]:
            p = param[_t, :].unsqueeze(0)
        else:
            p = param[torch.tensor([-1]).long(), :].unsqueeze(0)

        if param_key in ['alpha', 'beta', 'gamma', 'delta']:  # these params must be divided by population
            p = p / self.population

        return p  # shape 1 x s (or 1 it is broadcasted) x

    def differential_equations(self, t, x):
        """
        Returns the right-hand side of SIDARTHE model
        :param t: time t at which right-hand side is computed
        :param x: (Tensor) state of model at time t forall the fitted areas. S x #states (=8)
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
        p = {key: self.get_param_at_t(key, t) for key in self.params.keys()}

        S = x[:, 0]
        I = x[:, 1]
        D = x[:, 2]
        A = x[:, 3]
        R = x[:, 4]
        T = x[:, 5]

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
        ), dim=1).transpose(1,2).squeeze(0)

    def forward(self, time_grid) -> Dict:
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
        }

    def extend_param(self, value, length):
        """
        Extend each param in both dimensions (time, space) in case needed.
        :param value: a 2-d tensor, t|1 x s|1
        :param length: int temporal size to extend
        :return:
        """

        t_diff = length - value.shape[0]
        s_diff = self.n_areas - value.shape[1]


        ext_t_tensor = value[-1, :].reshape(1,-1).repeat((t_diff, 1))


        ext_t_tensor = torch.cat((value, ext_t_tensor), dim=0) # T x s

        ext_s_tensor = ext_t_tensor[:, -1].reshape(-1,1).repeat((1,s_diff))
        ext_tensor = torch.cat((ext_t_tensor, ext_s_tensor), dim=1)  # T x S

        rectified_param = ext_tensor
        return rectified_param
        # return torch.where(rectified_param >= EPS, rectified_param, rectified_param + EPS)