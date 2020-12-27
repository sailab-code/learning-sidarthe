import datetime
from pytorch_lightning import Callback
import numpy as np

from lcm.utils.visualization import generic_plot, Curve

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S"
]

DEFAULT_START_DATE = "2020-2-24"

class TensorboardLoggingCallback(Callback):
    def __init__(self):
        """
        Callback handling tensorboard visualizations of parameters
        and losses trends.
        """
        super().__init__()
        self.first_date = DEFAULT_START_DATE

    def on_fit_start(self, trainer, pl_module):
        self.first_date = trainer.dataset.first_date  # setting the actual start date of the outbreak

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        # plot params
        params_plots = self._plot_params_over_time(pl_module, trainer.dataset.region)
        for (plot, plot_title) in params_plots:
            trainer.logger.experiment.add_figure(f"{plot_title}", plot, close=True, global_step=-1)

    def _plot_params_over_time(self, pl_module, region, n_days=None):
        """
        Plots the model params
        :param pl_module: lightning module, the model
        :param region: list of regions
        :param n_days: (optional) number of days
        :return: a list of plots
        """
        param_plots = []
        if n_days is None:
            n_days = pl_module.beta.shape[0] # fixme remove completely

        # create the plots for the params over time, in groups of related rates
        for param_group, param_keys in pl_module.param_groups.items():
            params_subdict = {param_key: pl_module.params[param_key] for param_key in param_keys}
            for j in range(len(region)):
                for param_key, param in params_subdict.items():
                    param = pl_module.extend_param(param, n_days)
                    pl_x = list(range(n_days))
                    pl_title = f"{region[j]}/{param_group}/$\\{param_key}$ over time"
                    param_curve = Curve(pl_x, param[:,j].detach().numpy(), '-', f"$\\{param_key}$", color=None)
                    curves = [param_curve]

                    plot = generic_plot(curves, pl_title, None, formatter=self._format_xtick) #fixme set data from data
                    param_plots.append((plot, pl_title))

        return param_plots

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.plot_final_inferences(trainer, pl_module, outputs["hats"], batch)

    def plot_final_inferences(self, trainer, pl_module, hats, batch, prefix="forecast", collapse=False):
        """
        Plot inferences
        :param trainer:
        :param pl_module:
        :param hats: a Tensor with the predictions on the entire data
        :param batch: a Tensor with the entire batch of data
        :param prefix:
        :param collapse:
        :return:
        """

        inputs = batch[0]
        targets = {k:v.squeeze() for k,v in batch[1].items()}
        region = trainer.dataset.region

        # get normalized values
        population = pl_module.population
        norm_hats = self.normalize_values(hats, population)
        norm_targets = self.normalize_values(targets, population)

        # ranges for train/val/test
        dataset_size = inputs.shape[1]
        # validation on the next val_len days (or less if we have less data)
        train_size, val_size = trainer.dataset.train_size, trainer.dataset.val_size

        train_range = range(0, train_size)
        val_range = range(train_size, train_size+val_size)
        test_range = range(train_size+val_size, dataset_size)

        def get_curves(x_range, hat, target, key, color=None):
            pl_x = list(x_range)
            hat_curve = Curve(pl_x, hat, '-', label=f"Estimated {key.upper()}", color=color)
            if target is not None:
                target_curve = Curve(pl_x, target, '.', label=f"Actual {key.upper()}", color=color)
                return [hat_curve, target_curve]
            else:
                return [hat_curve]

        tot_curves = []
        for key in hats.keys():

            # skippable keys
            if key in ["sol"]:
                continue

            for j in range(len(region)):
                # separate keys that should be normalized to 1
                if key not in ["r0"]:
                    curr_hat_train = norm_hats[key][:train_size, j]
                    curr_hat_val = norm_hats[key][train_size:train_size+val_size, j]
                    curr_hat_test = norm_hats[key][train_size+val_size:, j]
                else:
                    curr_hat_train = hats[key][:train_size, j]
                    curr_hat_val = hats[key][train_size:train_size+val_size, j]
                    curr_hat_test = hats[key][train_size+val_size:, j]

                if key in targets:
                    # plot inf and target_t
                    target_train = norm_targets[key][:train_size, j]
                    target_val = norm_targets[key][train_size:train_size+val_size, j]
                    target_test = norm_targets[key][train_size+val_size:, j]
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

                pl_title = f"{key.upper()} - train/validation/test"
                fig = generic_plot(tot_curves, pl_title, None, formatter=self._format_xtick) #fixme set data from data
                trainer.logger.experiment.add_figure(f"{prefix}/{region[j]}/{key}", fig)

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

    def _format_xtick(self, n,v):
        start_date = self._parse_date(self.first_date[0]) # fixme first date [0]
        # def custom_xtick(n, v):
        return (start_date + datetime.timedelta(int(n))).strftime("%d %b")

        # return custom_xtick
