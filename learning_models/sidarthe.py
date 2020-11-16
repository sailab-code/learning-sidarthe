import json
import os
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from learning_models.abstract_model import AbstractModel
from learning_models.optimizers.new_sir_optimizer import NewSirOptimizer
from populations import populations
from torch_euler import Heun, euler
from utils import derivatives
from utils.visualization_utils import Curve, generic_plot, format_xtick, generate_format_xtick

EPS = 0


class Sidarthe(AbstractModel):
    dtype = torch.float32

    def __init__(self, parameters: Dict, population, init_cond, integrator, sample_time, **kwargs):
        super().__init__(init_cond, integrator, sample_time)
        self.set_initial_params(parameters)
        self.population = population
        self.model_name = kwargs.get("name", "sidarthe")

        self.target_weights = {
            "d": kwargs["d_weight"],
            "r": kwargs["r_weight"],
            "t": kwargs["t_weight"],
            "h": kwargs["h_weight"],
            "e": kwargs["e_weight"]
        }

        self.der_1st_reg = kwargs["der_1st_reg"]
        self.bound_reg = kwargs["bound_reg"]
        self.verbose = kwargs.get("verbose", False)
        self.loss_type = kwargs.get("loss_type", "nrmse")
        self.bound_loss_type = kwargs.get("bound_loss_type", "step")

        self.references = kwargs.get("references", None)
        self.targets = kwargs.get("targets", None)

        self.train_size = kwargs.get("train_size", None)
        self.val_size = kwargs.get("val_size", None)
        self.first_date = kwargs.get("first_date", None)

        if self.targets is not None:
            # compute normalization values
            averages = {
                key[0]: np.mean(value) if self.target_weights[key[0]] > 0.0 else 0.0 for key, value in self.targets.items()
            }
            max_average = np.max([value for value in averages.values()])
            self.norm_weights = {
                key: max_average / avg if avg > 0.0 else 0.0 for key, avg in averages.items()
            }
            print(self.norm_weights)

        if self.first_date is None:
            self.format_xtick = format_xtick
        else:
            self.format_xtick = generate_format_xtick(self.first_date)

    @classmethod
    def from_model_summaries(cls, model_path):
        with open(os.path.join(model_path, 'settings.json')) as settings_f:
            settings = json.load(settings_f)

        with open(os.path.join(model_path, 'final.json')) as final_f:
            final = json.load(final_f)

        # default value taken from csv
        init_conditions = settings.get('init_conditions', (59999576.0, 94, 94, 101, 101, 26, 7, 1))

        first_date = settings.get('first_date', "2020-02-24")

        return cls(
            parameters=final['params'],
            population=populations[settings['region']],
            init_cond=init_conditions,
            integrator=Heun if settings['integrator'] == 'Heun' else euler,
            sample_time=settings['t_inc'],
            first_date=first_date,
            d_weight=1., r_weight=1., t_weight=1., h_weight=1., e_weight=1.,
            der_1st_reg=0.,
            bound_reg=0.
        )

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
        }

    @property
    def param_groups(self) -> Dict:
        return {
            "infection_rates": ('alpha', 'beta', 'gamma', 'delta'),
            "detection_rates": ('epsilon', 'theta'),
            "symptoms_development_rates": ('eta', 'zeta'),
            "acute_symptoms_development_rates": ('mu', 'nu'),
            "recovery_rates": ('kappa', 'lambda', 'xi', 'rho', 'sigma'),
            "death_rates": tuple(['tau'])
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
        return self._params["delta"]
        # return self._params["beta"]

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
    def nu(self) -> torch.Tensor:
        return self._params["nu"]

    @property
    def tau(self) -> torch.Tensor:
        return self._params["tau"]

    @property
    def lambda_(self) -> torch.Tensor:
        return self._params["lambda"]

    @property
    def kappa(self) -> torch.Tensor:
        # return self._params["xi"]
        return self._params["kappa"]

    @property
    def zeta(self) -> torch.Tensor:
        return self._params["zeta"]
        # return self._params["eta"]

    @property
    def rho(self) -> torch.Tensor:
        return self._params["rho"]
        # return self._params["lambda"]

    @property
    def sigma(self) -> torch.Tensor:
        return self._params["sigma"]

    # endregion ModelParams
    @staticmethod
    def get_param_at_t(param, _t):
        _t = _t.long()
        if 0 <= _t < param.shape[0]:
            rectified_param = torch.relu(param[_t].unsqueeze(0))
        else:
            rectified_param = torch.relu(param[-1].unsqueeze(0).detach())
        
        return torch.where(rectified_param >= EPS, rectified_param, rectified_param + EPS)

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
        A_dot = zeta * I - (theta + mu + kappa) * A
        R_dot = eta * D + theta * A - (nu + xi) * R
        T_dot = mu * A + nu * R - (sigma + tau) * T
        E_dot = tau * T
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

    def omega(self, t):
        if t >= 0:
            return torch.tensor([self.init_cond[:8]], dtype=self.dtype)
        else:
            return torch.tensor([[1.] + [0.] * 7], dtype=self.dtype)

    def set_initial_params(self, params):
        self._params = {key: torch.tensor(value, dtype=self.dtype, requires_grad=True) for key, value in
                        params.items()}

    def set_params(self, params):
        self._params = params

    @staticmethod
    def __rmse_loss(target, hat):
        mask = torch.ge(target, 0)

        return torch.sqrt(
            0.5 * torch.mean(
                torch.pow(target[mask] - hat[mask], 2)
            )
        )

    @staticmethod
    def __mae_loss(target, hat):
        mask = torch.ge(target, 0)

        return torch.mean(
            torch.abs(target[mask] - hat[mask])
        )

    @staticmethod
    def __mape_loss(target, hat):
        mask = torch.ge(target, 0)

        return torch.mean(
            torch.abs(
                (target[mask] - hat[mask]) / target[mask]
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

    @classmethod
    def __loss_parameter_near_origin(cls, parameter: torch.Tensor, eps=1e-10):
        # apply ReLU
        rectified_param = torch.max(torch.full_like(parameter, eps), parameter)

        # apply loss
        return torch.pow(torch.log(rectified_param) / torch.log(torch.tensor(1. / 0.5e3)), 10)

    def first_derivative_loss(self):
        loss_1st_derivative_total = torch.zeros(1, dtype=self.dtype)
        if self.der_1st_reg != 0:
            for key, value in self.params.items():
                first_derivative = derivatives.first_derivative(value, self.time_step)
                # loss_1st_derivative_total = loss_1st_derivative_total + 0.5 * torch.abs(first_derivative, 2)
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
        if self.bound_reg != 0:

            for key, value in self._params.items():
                if self.bound_loss_type == "step":
                    bound_reg = self.__loss_lte_zero(value)
                elif self.bound_loss_type == "log":
                    bound_reg = self.__loss_parameter_near_origin(value)
                else:
                    raise Exception("Loss type not supported")

                bound_reg_total = bound_reg_total + bound_reg

        # average params bound regularization
        # bound_reg_total /= len(self.params)

        return self.bound_reg * torch.sum(bound_reg_total)

    def losses(self, inferences, targets) -> Dict:
        # this function converts target values to torch.tensor with specified dtype
        def to_torch_float(target):
            if isinstance(target, np.ndarray) or isinstance(target, list):
                return torch.tensor(target, dtype=self.dtype)
            return target.to(dtype=self.dtype)

        # uses to_torch_float to convert given targets
        targets = {key: to_torch_float(value) for key, value in targets.items()}

        def compute_total_loss(loss_function, weighted=True, normalized=False):
            losses = {
                key[0]: loss_function(targets[key], inferences[key]) for key, value in targets.items()
            }

            def weight_losses(losses_dict):
                return {
                    key: self.target_weights[key] * loss_v for key, loss_v in losses_dict.items()
                }

            def normalize_losses(losses_dict):
                return {
                    key: self.norm_weights[key] * loss_v for key, loss_v in losses_dict.items()
                }

            if weighted:
                losses = weight_losses(losses)
                if normalized:
                    losses = normalize_losses(losses)

            total_loss = sum([value for key, value in losses.items()])

            return total_loss, losses

        # compute losses

        # der_2nd_loss = self.second_derivative_loss()

        bound_reg = self.bound_parameter_regularization()

        if self.loss_type == "rmse":
            total_rmse, losses_dict = compute_total_loss(self.__rmse_loss)
            loss = torch.tensor([1e-4], dtype=self.dtype) * total_rmse
        elif self.loss_type == "mae":
            total_mae, losses_dict = compute_total_loss(self.__mae_loss)
            loss = total_mae
        elif self.loss_type == "mape":
            total_mape, losses_dict = compute_total_loss(self.__mape_loss)
            loss = total_mape
        elif self.loss_type == "nrmse":
            total_nrmse, losses_dict = compute_total_loss(self.__rmse_loss, normalized=True)
            loss = total_nrmse
        elif self.loss_type == "nmae":
            total_nmae, losses_dict = compute_total_loss(self.__mae_loss, normalized=True)
            loss = total_nmae
        else:
            raise ValueError(f"loss type {self.loss_type} not supported")

        der_1st_loss = self.first_derivative_loss()
        total_loss = loss + der_1st_loss + bound_reg
        val_loss, _ = compute_total_loss(self.__rmse_loss, weighted=False)

        return {
            self.val_loss_checked: val_loss.squeeze(0),
            self.backward_loss_key: total_loss.squeeze(0),
            "der_1st": der_1st_loss,
            "bound_reg": bound_reg,
            **losses_dict
        }

    def extend_param(self, value, length):
        len_diff = length - value.shape[0]
        ext_tensor = torch.tensor([value[-1] for _ in range(len_diff)], dtype=self.dtype)
        ext_tensor = torch.cat((value, ext_tensor))

        # return torch.relu(ext_tensor) + EPS
        rectified_param = torch.relu(ext_tensor)
        return torch.where(rectified_param >= EPS, rectified_param, rectified_param + EPS)

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
        c3 = extended_params['theta'] + extended_params['mu'] + extended_params['kappa']
        c4 = extended_params['nu'] + extended_params['xi']

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
    def init_trainable_model(cls, initial_params: dict, initial_conditions, targets, **model_params):
        model_cls = model_params["model_cls"]
        time_step = model_params["time_step"]
        population = model_params["population"]
        integrator = model_params["integrator"]

        d_weight = model_params.get("d_weight", 1.)
        r_weight = model_params.get("r_weight", 1.)
        t_weight = model_params.get("t_weight", 1.)
        h_weight = model_params.get("h_weight", 1.)
        e_weight = model_params.get("e_weight", 1.)

        der_1st_reg = model_params.get("der_1st_reg", 2e4)

        bound_reg = model_params.get("bound_reg", 1e5)

        verbose = model_params.get("verbose", False)

        loss_type = model_params.get("loss_type", "rmse")
        bound_loss_type = model_params.get("bound_loss_type", "step")

        references = model_params.get("references", None)
        train_size = model_params.get("train_size", None)
        val_size = model_params.get("val_size", None)
        first_date = model_params.get("first_date", None)

        return cls(initial_params, population, initial_conditions, integrator, time_step,
                   d_weight=d_weight,
                   r_weight=r_weight,
                   t_weight=t_weight,
                   h_weight=h_weight,
                   e_weight=e_weight,
                   der_1st_reg=der_1st_reg,
                   bound_reg=bound_reg,
                   verbose=verbose,
                   loss_type=loss_type,
                   references=references,
                   targets=targets,
                   train_size=train_size,
                   val_size=val_size,
                   first_date=first_date,
                   bound_loss_type=bound_loss_type
                   )

    @classmethod
    def init_optimizers(cls, model: 'Sidarthe', learning_rates: dict, optimizers_params: dict) -> List[Optimizer]:
        m = optimizers_params.get("m", 1 / 9)
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
        H0_detected = targets["h_detected"][0].item()  # dimessi guariti
        E0 = targets["e"][0].item()  # deceduti

        # for now we assume that the number of undetected is equal to the number of detected
        # meaning that half of the infectious were not detected
        I0 = D0  # isolamento domiciliare
        A0 = R0  # ricoverati con sintomi

        S0 = population - (I0 + D0 + A0 + R0 + T0 + H0_detected + E0)

        return (
            S0,
            I0,
            D0,
            A0,
            R0,
            T0,
            E0,
            H0_detected
        )

    def plot_params_over_time(self, n_days=None):
        param_plots = []

        if n_days is None:
            n_days = self.beta.shape[0]

        for param_group, param_keys in self.param_groups.items():
            params_subdict = {param_key: self.params[param_key] for param_key in param_keys}
            for param_key, param in params_subdict.items():
                param = self.extend_param(param, n_days)
                pl_x = list(range(n_days))
                pl_title = f"{param_group}/$\\{param_key}$ over time"
                param_curve = Curve(pl_x, param.detach().numpy(), '-', f"$\\{param_key}$", color=None)
                curves = [param_curve]

                if self.references is not None:
                    if param_key in self.references:
                        ref_curve = Curve(pl_x, self.references[param_key][:n_days], "--", f"$\\{param_key}$ reference",
                                          color=None)
                        curves.append(ref_curve)
                plot = generic_plot(curves, pl_title, None, formatter=self.format_xtick)
                param_plots.append((plot, pl_title))
        return param_plots

    @staticmethod
    def get_curves(x_range, hat, target, key, color=None):
        pl_x = list(x_range)
        hat_curve = Curve(pl_x, hat, '-', label=f"GF ({key})", color=color)
        if target is not None:
            target_curve = Curve(pl_x, target, '.', label=f"Data ({key})", color=color)
            return [hat_curve, target_curve]
        else:
            return [hat_curve]

    @staticmethod
    def normalize_values(values, norm):
        """normalize values by a norm, e.g. population"""
        return {key: np.array(value) / norm for key, value in values.items()}

    @staticmethod
    def slice_values(values, slice_):
        return {key: value[slice_] for key, value in values.items()}

    def plot_fits(self):
        fit_plots = []
        with torch.no_grad():

            targets = self.targets
            dataset_size = len(self.targets["d"])
            t_grid = torch.linspace(0, dataset_size, dataset_size + 1)

            inferences = self.inference(t_grid)
            norm_inferences = self.normalize_values(inferences, self.population)

            t_inc = self.time_step
            train_size = self.train_size
            val_size = self.val_size + train_size
            train_range = range(0, train_size)
            val_range = range(train_size, val_size)
            test_range = range(val_size, dataset_size)
            dataset_range = range(0, dataset_size)

            t_start = 0
            train_hat_slice = slice(t_start, int(train_size / t_inc), int(1 / t_inc))
            val_hat_slice = slice(int(train_size / t_inc), int(val_size / t_inc), int(1 / t_inc))
            test_hat_slice = slice(int(val_size / t_inc), int(dataset_size / t_inc), int(1 / t_inc))

            train_target_slice = slice(t_start, train_size, 1)
            val_target_slice = slice(train_size, val_size, 1)
            test_target_slice = slice(val_size, dataset_size, 1)
            dataset_target_slice = slice(t_start, dataset_size, 1)

            hat_train = self.slice_values(inferences, train_hat_slice)
            hat_val = self.slice_values(inferences, val_hat_slice)
            hat_test = self.slice_values(inferences, test_hat_slice)

            target_train = self.slice_values(targets, train_target_slice)
            target_val = self.slice_values(targets, val_target_slice)
            target_test = self.slice_values(targets, test_target_slice)

            norm_hat_train = self.normalize_values(hat_train, self.population)
            norm_hat_val = self.normalize_values(hat_val, self.population)
            norm_hat_test = self.normalize_values(hat_test, self.population)

            norm_target_train = self.normalize_values(target_train, self.population)
            norm_target_val = self.normalize_values(target_val, self.population)
            norm_target_test = self.normalize_values(target_test, self.population)

            for key in inferences.keys():
                if key in ["sol"]:
                    continue

                if key not in ["r0"]:
                    curr_hat_train = norm_hat_train[key]
                    curr_hat_val = norm_hat_val[key]
                    curr_hat_test = norm_hat_test[key]
                else:
                    curr_hat_train = hat_train[key]
                    curr_hat_val = hat_val[key]
                    curr_hat_test = hat_test[key]

                if key in norm_target_train:
                    target_train = norm_target_train[key]
                    target_val = norm_target_val[key]
                    target_test = norm_target_test[key]
                else:
                    target_train = None
                    target_val = None
                    target_test = None

                train_curves = self.get_curves(train_range, curr_hat_train, target_train, key, 'r')
                val_curves = self.get_curves(val_range, curr_hat_val, target_val, key, 'b')
                test_curves = self.get_curves(test_range, curr_hat_test, target_test, key, 'g')

                tot_curves = train_curves + val_curves + test_curves

                if self.references is not None:
                    reference_curve = Curve(list(dataset_range), self.references[key][dataset_target_slice], "--",
                                            label="Reference (Nature)")
                    tot_curves = tot_curves + [reference_curve]

                pl_title = f"{key.upper()} - train/validation/test/reference"
                fig = generic_plot(tot_curves, pl_title, None, formatter=self.format_xtick)
                pl_title = f"Estimated {key.upper()} on fit"
                fit_plots.append((fig, pl_title))

                if target_train is not None:
                    # add error plots
                    pl_title = f"{key.upper()} - errors"
                    fig_name = f"Error {key.upper()} on fit"
                    curr_errors_train = curr_hat_train - np.array(target_train)

                    curr_errors_val = curr_hat_val - np.array(target_val)

                    curr_errors_test = curr_hat_test - np.array(target_test)

                    train_curves = self.get_curves(train_range, curr_errors_train, None, key, 'r')
                    val_curves = self.get_curves(val_range, curr_errors_val, None, key, 'b')
                    test_curves = self.get_curves(test_range, curr_errors_test, None, key, 'g')
                    tot_curves = train_curves + val_curves + test_curves

                    fig = generic_plot(tot_curves, pl_title, None, formatter=self.format_xtick)
                    fit_plots.append((fig, fig_name))

        return fit_plots

    def plot_r0(self, r0):
        pl_x = list(range(0, r0.shape[0]))
        hat_curve = Curve(pl_x, r0.detach().numpy(), '-', label=f"Estimated R0", color=None)
        curves = [hat_curve]

        if self.references is not None:
            r0_slice = slice(0, r0.shape[0], 1)
            ref_curve = Curve(pl_x, self.references["r0"][r0_slice], "--", label=f"Reference R0", color=None)
            curves.append(ref_curve)

        pl_title = f"Estimated R0"
        plot = generic_plot(curves, pl_title, None, formatter=self.format_xtick)
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
                summary.add_figure(f"{fig_title}", fig, close=True, global_step=0)

            for fig, fig_title in self.plot_fits():
                summary.add_figure(f"fits/{fig_title}", fig, close=True, global_step=0)

    def log_info(self, epoch, losses, inferences, targets, summary: SummaryWriter = None):
        if self.verbose:
            print(f"Params at epoch {epoch}.")
            self.print_params()
            for key, value in self.params.items():
                print(f"{key}: {value.grad[0:2]}")

            print("\n")

        if summary is not None:
            for fig, fig_title in self.plot_params_over_time():
                summary.add_figure(f"{fig_title}", fig, close=True, global_step=epoch)

            for fig, fig_title in self.plot_fits():
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