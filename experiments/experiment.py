"""
Default abstract class to run a single experiment
"""

import os
from uuid import uuid4
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.report_utils import get_description, get_markdown_description
from utils.visualization_utils import generic_plot, Curve



class Experiment:
    def __init__(self,
                 region,
                 n_epochs,
                 time_step,
                 runs_directory="runs",
                 uuid=None,
                 uuid_prefix=None):

        self.runs_dir = runs_directory
        self.region = region
        self.time_step = time_step
        self.n_epochs = n_epochs

        self.initial_params = None
        self.dataset_params = None
        self.model_params = None
        self.learning_rates = None
        self.train_params = None
        self.loss_weights = None

        self.exp_path = None
        self.uuid_prefix = f"{uuid_prefix}_" if uuid_prefix is not None else ""
        self.uuid = uuid or uuid4()  # unique exp name, if not provided

        self.dataset = None
        self.model = None
        self.inferences = None
        self.references = None
        self.summary = None
        self.best_epoch = -1


    def set_exp_paths(self):
        """
        Creates experiment's folder and set exp_path attribute
        :return:
        """
        base_path = os.path.join(os.getcwd(), self.runs_dir)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        exp_path = os.path.join(base_path, self.model_params["name"])
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        # adds directory with the region name
        exp_path = os.path.join(exp_path, self.region)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        # adds directory with the uuid
        exp_path = os.path.join(exp_path, f"{self.uuid_prefix}{self.uuid}")
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        self.exp_path = exp_path

    @staticmethod
    def fill_missing_params(params, default_params):
        """
        Fill attributes of params that are missing with
        the default values.
        :param params: dict of parameters
        :param default_params: dict of default parameters
        :return: filled params dict
        """
        for k, v in default_params.items():
            if k not in params:
                params[k] = v

        return params

    def make_initial_params(self, **kwargs):
        return NotImplementedError

    def make_dataset_params(self, **kwargs):
        return NotImplementedError

    def make_model_params(self, **kwargs):
        return NotImplementedError

    def make_train_params(self, **kwargs):
        return NotImplementedError

    def make_loss_weights(self, **kwargs):
        return NotImplementedError

    def make_learning_rates(self, **kwargs):
        return NotImplementedError

    def make_references(self, **kwargs):
        return NotImplementedError

    def set_initial_params(self, initial_params):
        self.initial_params = initial_params

    def set_dataset_params(self, dataset_params):
        self.dataset_params = dataset_params

    def set_model_params(self, model_params, loss_weights):
        self.model_params = {**model_params, **loss_weights}

    def set_learning_rates(self, learning_rates):
        self.learning_rates = learning_rates

    def set_train_params(self, train_params):
        self.train_params = train_params

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def set_references(self, references):
        self.references = references

    def create_summaries(self):
        # tensorboard summary
        self.summary = SummaryWriter(f"{self.exp_path}")
        self.train_params["tensorboard_summary"] = self.summary

        # creates the json description file with all settings
        description = get_description(
            self.region, self.initial_params, self.learning_rates, self.loss_weights,
            self.dataset_params["train_size"], self.dataset_params["val_len"],
            self.model_params["der_1st_reg"], self.time_step,
            self.train_params["momentum"], self.train_params["m"], self.train_params["a"], self.model_params["loss_type"],
            self.model_params["integrator"], self.model_params["bound_reg"], self.model_params["bound_loss_type"], str(self.model_params["model_cls"])
        )

        json_description = json.dumps(description, indent=4)

        json_file = "settings.json"
        with open(os.path.join(self.exp_path, json_file), "a") as f:
            f.write(json_description)

        # pushes the html version of the summary on tensorboard
        self.summary.add_text("settings/summary", get_markdown_description(json_description, self.uuid))

        # region generate final report

    @staticmethod
    def normalize_values(values, norm):
        """normalize values by a norm, e.g. population"""
        return {key: np.array(value) / norm for key, value in values.items()}

    def compute_final_losses(self, x_target, targets):
        """
        Make inference on x_targets and compute the loss.
        :param x_target:
        :param targets:
        :return: Losses on train, val, test and all data sets,
        a tuple with predictions (hat_*) on each partition,
        a tuple with partitioned targets (target_*),
        dataset slice
        """
        dataset_size = len(x_target)
        # validation on the next val_len days (or less if we have less data)
        train_size, val_len = self.dataset_params["train_size"], self.dataset_params["val_len"]
        val_size = min(train_size + val_len, len(x_target) - 5)

        time_step = self.time_step

        t_grid = torch.linspace(0, dataset_size, int(dataset_size / time_step) + 1)

        self.inferences = self.model.inference(t_grid)

        # region data slices
        t_start = self.train_params["t_start"]
        train_hat_slice = slice(t_start, int(train_size / time_step), int(1 / time_step))
        val_hat_slice = slice(int(train_size / time_step), int(val_size / time_step), int(1 / time_step))
        test_hat_slice = slice(int(val_size / time_step), int(dataset_size / time_step), int(1 / time_step))
        dataset_hat_slice = slice(t_start, int(dataset_size / time_step), int(1 / time_step))

        train_target_slice = slice(t_start, train_size, 1)
        val_target_slice = slice(train_size, val_size, 1)
        test_target_slice = slice(val_size, dataset_size, 1)
        dataset_target_slice = slice(t_start, dataset_size, 1)
        # endregion

        # region slice inferences
        def slice_values(values, slice_):
            return {key: value[slice_] for key, value in values.items()}

        hat_train = slice_values(self.inferences, train_hat_slice)
        hat_val = slice_values(self.inferences, val_hat_slice)
        hat_test = slice_values(self.inferences, test_hat_slice)
        hat_dataset = slice_values(self.inferences, dataset_hat_slice)

        target_train = slice_values(targets, train_target_slice)
        target_val = slice_values(targets, val_target_slice)
        target_test = slice_values(targets, test_target_slice)
        target_dataset = slice_values(targets, dataset_target_slice)
        # endregion

        # region losses computation
        train_risks = self.model.losses(hat_train, target_train)
        val_risks = self.model.losses(hat_val, target_val)
        test_risks = self.model.losses(hat_test, target_test)
        dataset_risks = self.model.losses(hat_dataset, target_dataset)

        hat_t, target_t = (hat_train, hat_val, hat_test), (target_train, target_val, target_test)

        return {"train": train_risks, "val": val_risks, "test": test_risks, "dataset": dataset_risks}, hat_t, target_t, dataset_target_slice

    def valid_json_dict(self, tensor_dict):
        valid_dict = {}
        for key_, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                valid_dict[key_] = value.tolist()
            elif isinstance(value, dict):
                valid_dict[key_] = self.valid_json_dict(value)
            else:
                valid_dict[key_] = value
        return valid_dict

    def _make_final_report(self, risks):
        """
        Generate final report stored in JSON.
        :param risks: a dictionary with different risks values
        on different part of the data.
        :return: None
        """
        final_dict = {
            "best_epoch": self.best_epoch,
            "train_risks": risks["train"],
            "val_risks": risks["val"],
            "test_risks": risks["test"],
            "dataset_risks": risks["dataset"],
            "params": self.model.params,
        }

        json_final = json.dumps(self.valid_json_dict(final_dict), indent=4)
        json_file = "final.json"
        with open(os.path.join(self.exp_path, json_file), "a") as f:
            f.write(json_final)

        if self.summary:
            self.summary.add_text("settings/final", get_markdown_description(json_final, self.uuid))

    def _plot_final_params(self):
        # plot params
        params_plots = self.model.plot_params_over_time()
        for (plot, plot_title) in params_plots:
            self.summary.add_figure(f"final/{plot_title}", plot, close=True, global_step=-1)

    def plot_final_inferences(self, hat_t, target_t, dataset_target_slice, summary, prefix="final", collapse=False):
        """
        Plot inferences
        :param hat_t: a tuple with train val and test hat
        :param target_t: a tuple with train val and test target_t
        :param dataset_target_slice: data slice
        :param prefix:
        :return:
        """

        hat_train, hat_val, hat_test = hat_t
        target_train, target_val, target_test = target_t

        # get normalized values
        population = self.model_params["population"]
        norm_hat_train = self.normalize_values(hat_train, population)
        norm_hat_val = self.normalize_values(hat_val, population)
        norm_hat_test = self.normalize_values(hat_test, population)
        norm_target_train = self.normalize_values(target_train, population)
        norm_target_val = self.normalize_values(target_val, population)
        norm_target_test = self.normalize_values(target_test, population)

        # ranges for train/val/test
        dataset_size = len(self.dataset.inputs)
        # validation on the next val_len days (or less if we have less data)
        train_size, val_len = self.dataset.train_size, self.dataset.val_len
        val_size = min(train_size + val_len, dataset_size - 5)

        train_range = range(0, train_size)
        val_range = range(train_size, val_size)
        test_range = range(val_size, dataset_size)
        dataset_range = range(0, dataset_size)

        def get_curves(x_range, hat, target, key, color=None):
            pl_x = list(x_range)
            hat_curve = Curve(pl_x, hat, '-', label=f"Estimated {key.upper()}", color=color)
            if target is not None:
                target_curve = Curve(pl_x, target, '.', label=f"Actual {key.upper()}", color=color)
                return [hat_curve, target_curve]
            else:
                return [hat_curve]

        tot_curves = []
        for key in self.inferences.keys():

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

            if key in self.dataset.targets:
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

            # get reference in range of interest
            if self.references is not None:
                ref_y = self.references[key][dataset_target_slice]
                reference_curve = Curve(dataset_range, ref_y, "--", label="Reference (Nature)")
                tot_curves = tot_curves + [reference_curve]

            pl_title = f"{key.upper()} - train/validation/test/reference"
            fig = generic_plot(tot_curves, pl_title, None, formatter=self.model.format_xtick)
            summary.add_figure(f"{prefix}/{key}_global", fig)

    @staticmethod
    def get_configs_from_json(json_file):
        return json.load(open(json_file, "r"))

    def run_exp(self, **kwargs):
        """

        :param kwargs: Optional arguments. Expected (optional) attributes
            {'initial_params': dict,
            'model_params': dict,
            'dataset_params': dict,
            'train_params': dict,
            'learning_rates': dict,
            'loss_weights': dict,
            }

        :return: pretrained_model, uuid, results
        """
        # creates initial params
        initial_params = self.make_initial_params(**kwargs)
        self.set_initial_params(initial_params)

        # creates dataset params
        dataset_params = self.make_dataset_params(**kwargs)
        self.set_dataset_params(dataset_params)

        # gets the data for the pretrained_model
        dataset_cls = self.dataset_params["dataset_cls"]
        self.dataset = dataset_cls(self.dataset_params)
        self.dataset.make_dataset()
        x_targets = self.dataset.inputs
        targets = self.dataset.targets

        # create references
        references = self.make_references(**kwargs)
        self.set_references(references)

        # creates loss/train/learning_rates/pretrained_model params
        loss_weights = self.make_loss_weights(**kwargs)
        self.set_loss_weights(loss_weights)

        train_params = self.make_train_params(**kwargs)
        self.set_train_params(train_params)

        learning_rates = self.make_learning_rates(**kwargs)
        self.set_learning_rates(learning_rates)

        model_params = self.make_model_params(**kwargs)
        self.set_model_params(model_params, loss_weights)

        # creates experiment's folder
        self.set_exp_paths()

        # creates summaries
        self.create_summaries()

        # training
        model_cls = self.model_params["model_cls"]

        print(f"Running experiment {self.uuid}")

        self.model, logged_info, self.best_epoch = model_cls.train(
            targets,
            self.initial_params,
            self.learning_rates,
            self.n_epochs,
            self.model_params,
            **self.train_params
        )

        # inferences
        with torch.no_grad():
            risks, hat_t, target_t, dataset_slice = self.compute_final_losses(x_target=x_targets, targets=targets)
            self._make_final_report(risks)
            self._plot_final_params()
            self.plot_final_inferences(hat_t, target_t, dataset_slice, self.summary)

        self.summary.flush()

        return self.model, self.uuid, risks["val"][self.model.val_loss_checked]

    def days_before_diverge(self, rel_err, threshold=0.1):
        is_diverged_day = torch.gt(rel_err, threshold)
        diverged_days = torch.nonzero(is_diverged_day)
        n_diverged_days = diverged_days.shape[0]
        if diverged_days.shape[0] > 0:
            return diverged_days[0].item(), n_diverged_days
        else:
            return is_diverged_day.shape[0], n_diverged_days

    def eval_exp(self, threshold=0.1, **kwargs):
        """

                :param kwargs: Optional arguments. Expected (optional) attributes
                    {'initial_params': dict,
                    'model_params': dict,
                    'dataset_params': dict,
                    }

                :return: pretrained_model, uuid, results
                """
        # creates initial params
        initial_params = self.make_initial_params(**kwargs)
        self.set_initial_params(initial_params)

        # creates dataset params
        dataset_params = self.make_dataset_params(**kwargs)
        self.set_dataset_params(dataset_params)

        # gets the data for the pretrained_model
        dataset_cls = self.dataset_params["dataset_cls"]
        self.dataset = dataset_cls(self.dataset_params)
        self.dataset.make_dataset()
        x_targets = self.dataset.inputs
        targets = self.dataset.targets

        # create references
        references = self.make_references()
        self.set_references(references)

        # creates pretrained_model params
        model_params = self.make_model_params(**kwargs)
        self.set_model_params(model_params, {})

        # evaluation
        model_cls = self.model_params["model_cls"]
        initial_conditions = model_cls.compute_initial_conditions_from_targets(targets, model_params)

        train_params = self.make_train_params(**kwargs)
        self.set_train_params(train_params)

        self.model = model_cls.init_trainable_model(
            initial_params,
            initial_conditions,
            targets,
            **model_params
        )

        # creates experiment's folder
        self.set_exp_paths()

        dataset_size = len(x_targets)
        print(f"Dataset Size: {dataset_size}")
        time_step = 1.0
        t_grid = torch.linspace(0, dataset_size, int(dataset_size / time_step))  # NB removed +1
        self.inferences = self.model.inference(t_grid)

        with torch.no_grad():
            with open(os.path.join(self.exp_path, "val_rel_err.txt"), "w") as f:
                print(f"________VALIDATION________")
                val_slice = slice(self.dataset.train_size, self.dataset.train_size + self.dataset.val_len)
                for key in targets.keys():
                    print(f"________Relative Error of: {key}________")
                    target, hat = torch.tensor(targets[key][val_slice]), torch.tensor(self.inferences[key][val_slice])
                    mean_err, _, days_bef_div,n_diverged_days = self.get_n_days_before_diverges(target, hat, threshold)
                    f.write(f"{key} & {threshold} & {mean_err:.2f} & {days_bef_div} & {n_diverged_days}\\\\ \n")

            with open(os.path.join(self.exp_path, "test_rel_err.txt"), "w") as f:
                print(f"________TEST________")
                test_slice = slice(self.dataset.train_size + self.dataset.val_len, dataset_size)
                for key in targets.keys():
                    print(f"________Relative Error of: {key}________")
                    target, hat = torch.tensor(targets[key][test_slice]), torch.tensor(self.inferences[key][test_slice])
                    mean_err, _, days_bef_div, n_diverged_days = self.get_n_days_before_diverges(target, hat, threshold)
                    f.write(f"{key} & {threshold} & {mean_err:.2f} & {days_bef_div} & {n_diverged_days}\\\\ \n")

            # inferences
            with torch.no_grad():
                risks, hat_t, target_t, dataset_slice = self.compute_final_losses(x_target=x_targets, targets=targets)
                self._make_final_report(risks)

            r0 = np.max(self.inferences["r0"].numpy())
            return risks["val"]["nrmse"].item(), r0
            # return risks["val"]["nrmse"].item()

    def get_n_days_before_diverges(self, target, hat, threshold):
            mask = torch.gt(target, 0)
            rel_err = torch.abs((target[mask] - hat[mask]) / target[mask])
            print(target[mask])
            print(hat[mask])
            print(rel_err)
            mean_err = torch.mean(rel_err)
            std_err = torch.std(rel_err)
            print(f"Mean: {mean_err}; STD: {std_err}")
            day_before_diverge, n_diverged_days = self.days_before_diverge(rel_err, threshold=threshold)
            print(f"Diverges after {day_before_diverge} days")
            print(f"Number of days above threshold {n_diverged_days}")
            print("_____________________")
            return mean_err, std_err, day_before_diverge, n_diverged_days


