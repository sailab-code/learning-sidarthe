from collections import namedtuple
from typing import List, Dict

import numpy
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from learning_models.abstract_model import AbstractModel
from learning_models.new_sir_optimizer import NewSirOptimizer
from torch.optim import Adam
from utils import derivatives
from utils.visualization_utils import Curve, generic_plot, format_xtick


class Sidarthe(AbstractModel):
    dtype = torch.float64

    def __init__(self, parameters: Dict, population, init_cond, integrator, sample_time, **kwargs):
        super().__init__(init_cond, integrator, sample_time)
        self._params = {key: torch.tensor(value, dtype=self.dtype, requires_grad=True) for key, value in
                        parameters.items()}
        self.population = population

        self.d_weight = kwargs["d_weight"]
        self.r_weight = kwargs["r_weight"]
        self.t_weight = kwargs["t_weight"]
        self.h_weight = kwargs["h_weight"]
        self.e_weight = kwargs["e_weight"]
        self.der_1st_reg = kwargs["der_1st_reg"]
        self.bound_reg = kwargs["bound_reg"]
        self.verbose = kwargs["verbose"]
        self.loss_type = kwargs["loss_type"]

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
            "ni": self.ni,
            "tau": self.tau,
            "lambda": self.lambda_,
            "kappa": self.kappa,
            "zeta": self.zeta,
            "rho": self.rho,
            "sigma": self.sigma
        }

    # region ModelParams

    @property
    def alpha(self) -> torch.Tensor:
        return self._params["alpha"]

    @property
    def beta(self) -> torch.Tensor:
        return self._params["beta"]

    @property
    def gamma(self) -> torch.Tensor:
        return self._params["gamma"]

    @property
    def delta(self) -> torch.Tensor:
        #return self._params["delta"]
        return self._params["beta"]

    @property
    def epsilon(self) -> torch.Tensor:
        return self._params["epsilon"]

    @property
    def theta(self) -> torch.Tensor:
        return self._params["theta"]

    @property
    def xi(self) -> torch.Tensor:
        return self._params["xi"]

    @property
    def eta(self) -> torch.Tensor:
        return self._params["eta"]

    @property
    def mu(self) -> torch.Tensor:
        return self._params["mu"]

    @property
    def ni(self) -> torch.Tensor:
        return self._params["ni"]

    @property
    def tau(self) -> torch.Tensor:
        return self._params["tau"]

    @property
    def lambda_(self) -> torch.Tensor:
        return self._params["lambda"]

    @property
    def kappa(self) -> torch.Tensor:
        return self._params["xi"]
        #return self._params["kappa"]

    @property
    def zeta(self) -> torch.Tensor:
        #return self._params["zeta"]
        return self._params["eta"]

    @property
    def rho(self) -> torch.Tensor:
        return self._params["rho"]

    @property
    def sigma(self) -> torch.Tensor:
        return self._params["sigma"]

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
            _t = (_t / self.time_step).round().long()
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
        ni = get_param_at_t(self.ni, t)
        tau = get_param_at_t(self.tau, t)
        lambda_ = get_param_at_t(self.lambda_, t)
        kappa = get_param_at_t(self.kappa, t)
        zeta = get_param_at_t(self.zeta, t)
        rho = get_param_at_t(self.rho, t)
        sigma = get_param_at_t(self.sigma, t)
        # endregion parameters

        S = x[0]
        I = x[1]
        D = x[2]
        A = x[3]
        R = x[4]
        T = x[5]
        E = x[6]
        # H = x[7]

        # region equations

        S_dot = -S * (alpha * I + beta * D + gamma * A + delta * R)
        I_dot = -S_dot - (epsilon + zeta + lambda_) * I
        D_dot = epsilon * I - (eta + rho) * D
        A_dot = zeta * I - (theta + mu + kappa) * A
        R_dot = eta * D + theta * A - (ni + xi) * R
        T_dot = mu * A + ni * R - (sigma + tau) * T
        E_dot = tau * T
        # H_dot = lambda_ * I + rho * D + kappa * A + zeta * R + sigma * T

        # endregion equations

        return torch.cat((
            S_dot,
            I_dot,
            D_dot,
            A_dot,
            R_dot,
            T_dot,
            E_dot
            # H_dot,
        ), dim=0)

    def omega(self, t):
        if t >= 0:
            return torch.tensor([self.init_cond[:7]], dtype=self.dtype)
        else:
            return torch.tensor([[1.] + [0.] * 6], dtype=self.dtype)

    def set_params(self, params):
        self._params = params

    @staticmethod
    def __rmse_loss(target, hat):
        return torch.sqrt(
            0.5 * torch.mean(
                torch.pow(target - hat, 2)
            )
        )

    @staticmethod
    def __mape_loss(target, hat):
        return torch.mean(
            torch.abs(
                (target - hat) / target
            )
        )

    @classmethod
    def __loss_gte_one(cls, parameter):
        return parameter.abs() * torch.where(parameter.ge(1.),
                                             torch.ones(1, dtype=cls.dtype),
                                             torch.zeros(1, dtype=cls.dtype))

    @classmethod
    def __loss_lte_zero(cls, parameter: torch.Tensor):
        return parameter.abs() * torch.where(parameter.le(0.),
                                             torch.ones(1, dtype=cls.dtype),
                                             torch.zeros(1, dtype=cls.dtype))

    def first_derivative_loss(self):
        loss_1st_derivative_total = torch.zeros(1, dtype=self.dtype)
        if self.der_1st_reg != 0:
            for key, value in self.params.items():
                first_derivative = derivatives.first_derivative(value, self.time_step)
                loss_1st_derivative_total = loss_1st_derivative_total + 0.5 * torch.pow(first_derivative, 2)

        return self.der_1st_reg * torch.mean(loss_1st_derivative_total)

    def second_derivative_loss(self):
        loss_2nd_derivative_total = torch.zeros(1, dtype=self.dtype)
        for key, value in self.params.items():
            second_derivative = derivatives.second_derivative(value, self.time_step)
            loss_2nd_derivative_total = loss_2nd_derivative_total + 0.5 * torch.pow(second_derivative, 2)

        return torch.mean(loss_2nd_derivative_total)

    def bound_parameter_regularization(self):
        bound_reg_total = torch.zeros(1, dtype=self.dtype)
        for key, value in self.params.items():
            bound_reg = self.__loss_gte_one(value) + self.__loss_lte_zero(value)
            bound_reg_total = bound_reg_total + bound_reg

        return self.bound_reg * torch.mean(bound_reg_total)

    def losses(self, inferences, targets) -> Dict:

        # this function converts target values to torch.tensor with specified dtype
        def to_torch_float(target):
            if isinstance(target, numpy.ndarray) or isinstance(target, list):
                return torch.tensor(target, dtype=self.dtype)
            return target.to(dtype=self.dtype)

        # uses to_torch_float to convert given targets
        targets = {key: to_torch_float(value) for key, value in targets.items()}

        d_rmse_loss = self.__rmse_loss(targets["d"], inferences["d"])
        r_rmse_loss = self.__rmse_loss(targets["r"], inferences["r"])
        t_rmse_loss = self.__rmse_loss(targets["t"], inferences["t"])
        h_rmse_loss = self.__rmse_loss(targets["h_detected"], inferences["h_detected"])
        e_rmse_loss = self.__rmse_loss(targets["e"], inferences["e"])

        d_mape_loss = self.__mape_loss(targets["d"], inferences["d"])
        r_mape_loss = self.__mape_loss(targets["r"], inferences["r"])
        t_mape_loss = self.__mape_loss(targets["t"], inferences["t"])
        h_mape_loss = self.__mape_loss(targets["h_detected"], inferences["h_detected"])
        e_mape_loss = self.__mape_loss(targets["e"], inferences["e"])

        total_rmse = self.d_weight * d_rmse_loss + self.r_weight * r_rmse_loss + self.t_weight * t_rmse_loss \
                     + self.h_weight * h_rmse_loss + self.e_weight * e_rmse_loss

        total_mape = self.d_weight * d_mape_loss + self.r_weight * r_mape_loss + self.t_weight * t_mape_loss \
                     + self.h_weight * h_mape_loss + self.e_weight * e_mape_loss

        der_1st_loss = self.first_derivative_loss()
        der_2nd_loss = self.second_derivative_loss()

        bound_reg = self.bound_parameter_regularization()

        if self.loss_type is "rmse":
            loss = total_rmse
        elif self.loss_type is "mape":
            loss = total_mape
        else:
            raise ValueError(f"loss type {self.loss_type} not supported")

        total_loss = loss + der_1st_loss + bound_reg

        return {
            self.val_loss_checked: loss,
            "d_rmse": d_rmse_loss,
            "r_rmse": r_rmse_loss,
            "t_rmse": t_rmse_loss,
            "h_rmse": h_rmse_loss,
            "e_rmse": e_rmse_loss,
            "d_mape": d_mape_loss,
            "r_mape": r_mape_loss,
            "t_mape": t_mape_loss,
            "h_mape": h_mape_loss,
            "e_mape": e_mape_loss,
            "der_1st": der_1st_loss,
            #"der_2nd": der_2nd_loss,
            "bound_reg": bound_reg,
            self.backward_loss_key: total_loss
        }

    @staticmethod
    def extend_param(value, length):
        len_diff = length - value.shape[0]
        if len_diff > 0:
            return torch.cat((value, value[-1].expand(len_diff)))
        else:
            return value

    def inference(self, time_grid) -> Dict:
        sol = self.integrate(time_grid)
        s = sol[:, 0]
        i = sol[:, 1]
        d = sol[:, 2]
        a = sol[:, 3]
        r = sol[:, 4]
        t = sol[:, 5]
        e = sol[:, 6]
        h = self.population - (s + i + d + a + r + t + e)

        extended_params = {key: self.extend_param(value, time_grid.shape[0]) for key, value in self.params.items()}

        h_detected = self.init_cond[6] + torch.cumsum(
            extended_params['rho'] * d + extended_params['xi'] * r + extended_params['sigma'] * t,
            dim=0
        )

        # region compute R0
        c1 = extended_params['epsilon'] + extended_params['zeta'] + extended_params['lambda']
        c2 = extended_params['eta'] + extended_params['rho']
        c3 = extended_params['theta'] + extended_params['mu'] + extended_params['kappa']
        c4 = extended_params['ni'] + extended_params['xi']

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

    @classmethod
    def init_trainable_model(cls, initial_params: dict, initial_conditions, **model_params):
        time_step = model_params["time_step"]
        population = model_params["population"]
        integrator = model_params["integrator"]

        d_weight = model_params.get("d_weight", 0.)
        r_weight = model_params.get("r_weight", 0.)
        t_weight = model_params.get("t_weight", 0.)
        h_weight = model_params.get("h_weight", 0.)
        e_weight = model_params.get("e_weight", 1.)

        der_1st_reg = model_params.get("der_1st_reg", 2e4)

        bound_reg = model_params.get("bound_reg", 1e5)

        verbose = model_params.get("verbose", False)

        loss_type = model_params.get("loss_type", "rmse")

        return Sidarthe(initial_params, population, initial_conditions, integrator, time_step,
                        d_weight=d_weight,
                        r_weight=r_weight,
                        t_weight=t_weight,
                        h_weight=h_weight,
                        e_weight=e_weight,
                        der_1st_reg=der_1st_reg,
                        bound_reg=bound_reg,
                        verbose=verbose,
                        loss_type=loss_type
                        )

    @classmethod
    def init_optimizers(cls, model: 'Sidarthe', learning_rates: dict, optimizers_params: dict) -> List[Optimizer]:
        m = optimizers_params.get("m", 1/9)
        a = optimizers_params.get("a", 0.05)
        momentum = optimizers_params.get("momentum", True)
        summary = optimizers_params.get("tensorboard_summary", None)

        optimizer = NewSirOptimizer(model._params, learning_rates, m=m, a=a, momentum=momentum, summary=summary)
        return [optimizer]

    @classmethod
    def update_optimizers(cls, optimizers, model, learning_rates: dict):
        pass

    @staticmethod
    def compute_initial_conditions_from_targets(targets: dict, model_params: dict):

        """
        targets = {
            "d": "isolamento_domiciliare",
            "r": "ricoverati_con_sintomi",
            "t": "terapia_intensiva",
            "h_detected": "dimessi_guariti",
            "e": "deceduti"
        }
        """

        population = model_params["population"]

        D0 = targets["d"][0].item()  # isolamento
        R0 = targets["r"][0].item()  # ricoverati con sintomi
        T0 = targets["t"][0].item()  # terapia intensiva
        H0 = targets["h_detected"][0].item()  # dimessi guariti
        E0 = targets["e"][0].item()  # deceduti

        # for now we assume that the number of undetected is equal to the number of detected
        # meaning that half of the infectious were not detected
        I0 = D0  # isolamento domiciliare
        A0 = R0  # ricoverati con sintomi

        # we could consider them to be 0, alternatively
        #I0 = 0.  # isolamento domiciliare
        #A0 = 0.  # ricoverati con sintomi
        # TODO: maybe there are better options?

        S0 = population - (I0 + D0 + A0 + R0 + T0 + H0 + E0)

        return (
            S0,
            I0,
            D0,
            A0,
            R0,
            T0,
            E0,
            H0
        )

    def plot_params_over_time(self):
        param_plots = []
        max_len = max([value.shape[0] for key, value in self.params.items()])
        for key, value in self.params.items():
            value = self.extend_param(value, max_len)
            size = self.alpha.shape[0]
            pl_x = list(range(size))
            pl_title = f"$\\{key}$ over time"
            param_curve = Curve(pl_x, value.detach().numpy(), '-', f"$\\{key}$", color=None)
            plot = generic_plot([param_curve], pl_title, None, formatter=format_xtick)
            param_plots.append((plot, pl_title))

        return param_plots

    def plot_fits(self, inferences, targets):
        fit_plots = []
        for key, target in targets.items():
            inference = inferences[key]
            pl_x = list(range(0, target.shape[0]))
            hat_curve = Curve(pl_x, inference.detach().numpy(), '-', label=f"Estimated {key.upper()}", color=None)
            target_curve = Curve(pl_x, target, '.', label=f"Actual {key.upper()}", color=None)
            curves = [hat_curve, target_curve]
            pl_title = f"Estimated {key.upper()} on fit"
            plot = generic_plot(curves, pl_title, None, formatter=format_xtick)
            fit_plots.append((plot, pl_title))

        return fit_plots

    def plot_r0(self, r0):
        pl_x = list(range(0, r0.shape[0]))
        hat_curve = Curve(pl_x, r0.detach().numpy(), '-', label=f"Estimated R0", color=None)
        pl_title = f"Estimated R0"
        plot = generic_plot([hat_curve], pl_title, None, formatter=format_xtick)
        return (plot, pl_title)

    def print_params(self):
        for key, value in self.params.items():
            print(f"{key}: {value.detach().numpy()}")

    def log_initial_info(self, summary: SummaryWriter):
        if self.verbose:
            print("Initial params")
            self.print_params()
            print("\n")

        if summary is not None:
            for fig, fig_title in self.plot_params_over_time():
                summary.add_figure(f"params_over_time/{fig_title}", fig, close=True, global_step=0)

    def log_info(self, epoch, losses, inferences, targets, summary: SummaryWriter = None):

        if self.verbose:
            print(f"Params at epoch {epoch}.")
            self.print_params()
            for key, value in self.params.items():
                print(f"{key}: {value.grad[0:2]}")

            print("\n")

        if summary is not None:
            for fig, fig_title in self.plot_params_over_time():
                summary.add_figure(f"params_over_time/{fig_title}", fig, close=True, global_step=epoch)

            for fig, fig_title in self.plot_fits(inferences, targets):
                summary.add_figure(f"fits/{fig_title}", fig, close=True, global_step=epoch)

            r0_plot, r0_pl_title = self.plot_r0(inferences["r0"])
            summary.add_figure(f"fits/{r0_pl_title}", r0_plot, close=True, global_step=epoch)

            for key, value in losses.items():
                summary.add_scalar(f"losses/{key}", value.detach().numpy(), global_step=epoch)

            """for key, value in self.params.items():
                if value.grad is not None:
                    summary.add_scalar(f"grads/{key}", value.grad.detach().norm(), global_step=epoch)"""

        detached_losses = {key: value.detach().numpy() for key, value in losses.items()}

        return {
            "epoch": epoch,
            "losses": detached_losses
        }

    def log_validation_error(self, epoch, val_losses, summary: SummaryWriter = None):
        if summary is not None:
            summary.add_scalar("losses/validation_loss", val_losses[self.val_loss_checked], global_step=epoch)

    def regularize_gradients(self):
        for key, value in self.params.items():
            torch.nn.utils.clip_grad_norm_(value, 20.)

    @property
    def val_loss_checked(self):
        return self.loss_type
