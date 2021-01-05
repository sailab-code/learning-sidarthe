from typing import Dict
import torch

from lcm.sidarthe_extended import SidartheExtended


class SidartheMobility(SidartheExtended):
    #    dtype = torch.float32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get("name", "sidarthe_mobility")

        # filter mobility from first date onward
        self.mobility = kwargs["mobility"]

    @property
    def params(self) -> Dict:
        params = super().params
        params["mobility_0"] = self.mobility_0

        return params

    @property
    def mobility_0(self) -> torch.Tensor:
        return self._params["mobility_0"]

    def differential_equations(self, t, x):
        """
        Returns the right-hand side of SIDARTHE model enriched with Mobility
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
        :return: right-hand side of Mobility SIDARTHE model, i.e. f(t,:,x(t)), the second dimension corresponds to the areas to fit.
        """

        p = {key: self.get_param_at_t(key, t) for key in self.params.keys()}

        S = x[:,0]
        I = x[:,1]
        D = x[:,2]
        A = x[:,3]
        R = x[:,4]
        T = x[:,5]


        mobility = self.mobility.iloc[int(t)]  # t.item()

        S_dot = -S * (p['mobility_0'] * mobility * (p['alpha'] * I + p['beta'] * D) + p['gamma'] * A + p['delta'] * R)
        I_dot = -S_dot - (p['epsilon'] + p['zeta'] + p['lambda']) * I
        D_dot = p['epsilon'] * I - (p['eta'] + p['rho']) * D
        A_dot = p['zeta'] * I - (p['theta'] + p['mu'] + p['kappa']) * A
        R_dot = p['eta'] * D + p['theta'] * A - (p['nu'] + p['xi']) * R
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
        ), dim=1).transpose(1, 2).squeeze(0)
