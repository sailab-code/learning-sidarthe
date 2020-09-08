import abc
import time
from typing import List, Dict

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter


class AbstractModel(metaclass=abc.ABCMeta):
    dtype=torch.float64

    def __init__(self, init_cond, integrator, sample_time = 1.):
        """
        Abstract model with integration
        :param params: dictionary containing the parameter name and its initial value
        :param init_cond: list containing the initial values for the model states.
                        Must be same length of the return of differential_equations
        :param integrator: integrator method. Receives time t and current value x(t) where x is the state
        """
        self.init_cond = init_cond
        self.integrator = integrator
        self.time_step = sample_time

    @property
    @abc.abstractmethod
    def params(self) -> Dict:
        pass

    @params.setter
    @abc.abstractmethod
    def params(self, value: Dict):
        pass

    @abc.abstractmethod
    def differential_equations(self, t, x):
        pass

    @abc.abstractmethod
    def omega(self, t):
        pass

    @abc.abstractmethod
    def losses(self, inferences, targets) -> Dict:
        pass

    def integrate(self, time_grid):
        return self.integrator(self.differential_equations, self.omega, time_grid)

    @abc.abstractmethod
    def inference(self, time_grid) -> Dict:
        pass;

    @property
    def trainable_parameters(self):
        return [value for key, value in self.params.items()]

    @abc.abstractmethod
    def set_params(self, params):
        pass

    @classmethod
    @abc.abstractmethod
    def init_trainable_model(cls, initial_params: dict, initial_conditions, integrator, targets, **model_params):
        """
        Returns the initialized model

        :param initial_params: initial values for the parameters
        :param initial_conditions: initial conditions for the model
        :param integrator: integrator to use
        :param targets: targets to fit
        :param *
        :return: the initialized model
        """

    @classmethod
    @abc.abstractmethod
    def init_optimizers(cls, model, learning_rates: dict, optimizers_params: dict) -> List[Optimizer]:
        pass

    @classmethod
    @abc.abstractmethod
    def update_optimizers(cls, optimizers, model, learning_rates: dict):
        pass

    def regularize_gradients(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def compute_initial_conditions_from_targets(targets: dict, model_params: dict):
        pass

    def log_initial_info(self, summary: SummaryWriter):
        pass

    def log_info(self, epoch, losses, inferences, targets, summary: SummaryWriter = None):
        return {
            "epoch": epoch,
            "mse": losses[self.val_loss_checked]
        }

    def log_time_per_epoch(self, epoch, time_per_epoch, summary: SummaryWriter = None):
        pass

    def log_validation_error(self, epoch, val_losses, summary: SummaryWriter = None):
        pass

    def print_model_info(self):
        pass

    @property
    def val_loss_checked(self):
        return "mse"

    @property
    def backward_loss_key(self):
        return "backward"

    @classmethod
    def train(cls, targets: dict, initial_params: dict,
              learning_rates: dict,
              n_epochs,
              model_params: dict,
              **train_params):

        log_epoch_steps = train_params.get("log_epoch_steps", 50)
        validation_epoch_steps = train_params.get("validation_epoch_steps", 10)
        summary = train_params.get("tensorboard_summary", None)

        t_start = train_params["t_start"]
        t_end = train_params["t_end"]
        time_step = train_params["time_step"]
        val_size = train_params["val_size"]

        train_steps = int((t_end - t_start) / time_step)
        train_time_grid = torch.linspace(t_start, t_end, train_steps + 1)
        train_target_slice = slice(t_start, t_end, 1)
        train_hat_slice = slice(int(t_start / time_step), int(t_end / time_step), int(1 / time_step))

        val_steps = int((t_end + val_size - t_start) / time_step)
        val_time_grid = torch.linspace(t_start, t_end + val_size, val_steps + 1)
        val_target_slice = slice(t_end, t_end + val_size, 1)
        val_hat_slice = slice(int(t_end / time_step), int((t_end + val_size) / time_step), int(1 / time_step))

        def to_torch_sliced_tensor(value, slice):
            if value is None:
                return None
            else:
                return torch.tensor(value[slice])

        train_targets = {key: to_torch_sliced_tensor(value, train_target_slice) for key, value in targets.items()}
        val_targets = {key: to_torch_sliced_tensor(value, val_target_slice) for key, value in targets.items()}

        initial_conditions = cls.compute_initial_conditions_from_targets(train_targets, model_params)

        model: AbstractModel = cls.init_trainable_model(initial_params,
                                                        initial_conditions, targets, **model_params)

        optimizers = cls.init_optimizers(model, learning_rates, train_params)

        model.log_initial_info(summary)

        # early stopping stuff
        early_stopping_conf = train_params.get("early_stopping", {})
        best = torch.tensor(1e12, dtype=cls.dtype)
        best_epoch = -1
        best_params = model.params
        patience = 0
        n_lr_updates = 0
        max_no_improve = early_stopping_conf.get("max_no_improve", 25)
        max_n_lr_updates = early_stopping_conf.get("max_n_lr_updates", 5)
        lr_fraction = early_stopping_conf.get("lr_fraction", 2.)

        logged_info: List[Dict] = []

        inferences = model.inference(train_time_grid)
        print(f'Initial R0: {inferences["r0"]}')
        time_start = time.time()
        for epoch in range(1, n_epochs + 1):
            for optimizer in optimizers:
                optimizer.zero_grad()

            inferences = model.inference(train_time_grid)
            train_hats = {key: value[train_hat_slice] for key, value in inferences.items()}

            losses = model.losses(train_hats, train_targets)
            losses[model.backward_loss_key].backward()

            model.regularize_gradients()

            for optimizer in optimizers:
                optimizer.step()

            if epoch % log_epoch_steps == 0:
                print(f"epoch {epoch} / {n_epochs}")
                log_info = model.log_info(epoch, losses, train_hats, train_targets, summary)
                logged_info.append(log_info)
                epoch_steps_measure = time.time() - time_start
                time_per_epoch = epoch_steps_measure / log_epoch_steps
                model.log_time_per_epoch(epoch, time_per_epoch)
                print("Average time for epoch: {}".format(time_per_epoch))
                time_start = time.time()

            if epoch % validation_epoch_steps == 0:
                with torch.no_grad():
                    val_inferences = model.inference(val_time_grid)
                    val_hats = {key: value[val_hat_slice] for key, value in val_inferences.items()}
                    val_losses = model.losses(val_hats, val_targets)
                    model.log_validation_error(epoch, val_losses, summary)
                val_loss = val_losses[model.val_loss_checked]
                if val_loss < best and not torch.isclose(val_loss, best):
                    # maintains the best solution found so far
                    best = val_loss
                    best_params = model.params
                    best_epoch = {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "losses": losses
                    }
                    patience = 0
                elif patience < max_no_improve:
                    patience += 1
                elif n_lr_updates < max_n_lr_updates:
                    # when patience is over reduce learning rate by lr_fraction
                    print(f"Reducing learning rate at step {epoch}")
                    learning_rates = {key: value / lr_fraction for key, value in learning_rates.items()}
                    cls.update_optimizers(optimizers, model, learning_rates)
                    n_lr_updates += 1
                    patience = 0
                else:
                    print(f"Early stop at step {epoch}")
                    break

        model.set_params(best_params)
        print("-" * 20)
        print("Best: " + str(best))
        model.print_model_info()
        print("\n")

        return model, logged_info, best_epoch
