import os
import datetime
from pytorch_lightning import Callback
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils.visualization_utils import generic_plot, Curve #fixme wrong path

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S"
]


class TensorboardLoggingCallback(Callback):
    def __init__(self):
        """
        Callback handling tensorboard visualizations of parameters
        and losses trends.
        """

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        # plot params
        params_plots = self._plot_params_over_time(pl_module)
        for (plot, plot_title) in params_plots:
            # self.summary.add_figure(f"final/{plot_title}", plot, close=True, global_step=-1)
            trainer.logger.experiment.add_figure(f"final/{plot_title}", plot, close=True, global_step=-1)

    def _plot_params_over_time(self, pl_module, n_days=None):
        """
        Plots the model params
        :param pl_module: lightning module, the model
        :param n_days: (optional) number of days
        :return: a list of plots
        """
        param_plots = []
        if n_days is None:
            n_days = pl_module.beta.shape[0] # fixme remove completely

        # create the plots for the params over time, in groups of related rates
        for param_group, param_keys in pl_module.param_groups.items():
            params_subdict = {param_key: pl_module.params[param_key] for param_key in param_keys}
            for param_key, param in params_subdict.items():
                param = pl_module.extend_param(param, n_days)
                pl_x = list(range(n_days))
                pl_title = f"{param_group}/$\\{param_key}$ over time"
                param_curve = Curve(pl_x, param.detach().numpy(), '-', f"$\\{param_key}$", color=None)
                curves = [param_curve]

                plot = generic_plot(curves, pl_title, None, formatter=self._format_xtick("2020-02-24"))
                param_plots.append((plot, pl_title))

        return param_plots

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.plot_final_inferences(trainer, pl_module, outputs, batch)

    def plot_final_inferences(self, trainer, pl_module, hat_t, target_t, prefix="final", collapse=False):
        """
        Plot inferences
        :param hat_t: a tuple with train val and test hat
        :param target_t: a tuple with train val and test target_t
        :param prefix:
        :return:
        """

        # fixme
        hat_train, hat_val, hat_test = hat_t
        target_train, target_val, target_test = target_t

        # get normalized values
        population = pl_module["population"]
        norm_hat_train = self.normalize_values(hat_train, population)
        norm_hat_val = self.normalize_values(hat_val, population)
        norm_hat_test = self.normalize_values(hat_test, population)
        norm_target_train = self.normalize_values(target_train, population)
        norm_target_val = self.normalize_values(target_val, population)
        norm_target_test = self.normalize_values(target_test, population)

        # ranges for train/val/test
        dataset_size = len(target_t.inputs)
        # validation on the next val_len days (or less if we have less data)
        train_size, val_len = trainer.train_size, trainer.val_size
        val_size = min(train_size + val_len, dataset_size - 5)

        train_range = range(0, train_size)
        val_range = range(train_size, val_size)
        test_range = range(val_size, dataset_size)

        def get_curves(x_range, hat, target, key, color=None):
            pl_x = list(x_range)
            hat_curve = Curve(pl_x, hat, '-', label=f"Estimated {key.upper()}", color=color)
            if target is not None:
                target_curve = Curve(pl_x, target, '.', label=f"Actual {key.upper()}", color=color)
                return [hat_curve, target_curve]
            else:
                return [hat_curve]

        tot_curves = []
        for key in hat_t.keys():

            # skippable keys
            if key in ["sol"]:
                continue

            # separate keys that should be normalized to 1
            if key not in ["r0"]:
                curr_hat_train = norm_hat_train[key]
                curr_hat_val = norm_hat_val[key]
                curr_hat_test = norm_hat_test[key]
            else:
                curr_hat_train = hat_train[key]
                curr_hat_val = hat_val[key]
                curr_hat_test = hat_test[key]

            if key in hat_t:
                # plot inf and target_t
                target_train = norm_target_train[key]
                target_val = norm_target_val[key]
                target_test = norm_target_test[key]
                pass
            else:
                target_train = None
                target_val = None
                target_test = None
                pass

            train_curves = get_curves(train_range, curr_hat_train, target_train, key, 'r')
            val_curves = get_curves(val_range, curr_hat_val, target_val, key, 'b')
            test_curves = get_curves(test_range, curr_hat_test, target_test, key, 'g')

            if collapse:
                tot_curves += train_curves + val_curves + test_curves
            else:
                tot_curves = train_curves + val_curves + test_curves

            pl_title = f"{key.upper()} - train/validation/test/reference"
            fig = generic_plot(tot_curves, pl_title, None, formatter=self._format_xtick)
            trainer.logger.experiment.add_figure(f"{prefix}/{key}_global", fig)

    @staticmethod
    def normalize_values(values, norm):
        """normalize values by a norm, e.g. population"""
        return {key: np.array(value) / norm for key, value in values.items()}

    @staticmethod
    def _parse_date(date):
        for date_format in DATE_FORMATS:
            try:
                return datetime.datetime.strptime(date, date_format)
            except ValueError:
                continue

        raise ValueError("No date formats were able to parse date")

    def _format_xtick(self, start_date):
        start_date = self._parse_date(start_date)

        def custom_xtick(n, v):
            return (start_date + datetime.timedelta(int(n))).strftime("%d %b")

        return custom_xtick
