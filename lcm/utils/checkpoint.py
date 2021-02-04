from typing import List, Dict, Union, TextIO
import json

from lcm.datasets.sir_dataset import SirDataModule
from lcm.datasets.sidarthe_dataset import SidartheDataModule
from lcm.datasets.st_sidarthe_dataset import SpatioTemporalSidartheDataset

from lcm.integrators.fixed_step import *
from lcm.losses import compose_losses
from lcm.losses.regularization_losses import *

from lcm.losses.target_losses import *
from lcm.st_sidarthe import SpatioTemporalSidarthe


def load_from_json_checkpoint(checkpoint: Union[str, TextIO]):
    if isinstance(checkpoint, str):
        with open(checkpoint, "r") as checkpoint_file:
            return _load_from_checkpoint(json.load(checkpoint_file))
    elif isinstance(checkpoint, TextIO):
        return _load_from_checkpoint(json.load(checkpoint))

def _load_from_checkpoint(checkpoint: Dict):
    params = checkpoint['params']
    settings = checkpoint['settings']
    model = settings["model"]
    dataset = load_dataset(settings["dataset"])

    return model, dataset


def load_model(model: Dict):
    class_ = model["class"]
    initial_conditions = model["initial_conditions"]
    integrator = load_integrator(model["integrator"])
    time_step = model["time_step"]
    eps = model["EPS"]
    params = model["params"]
    lrates = model["learning_rates"]
    momentum_settings = model["momentum_settings"]
    population = model["population"]
    loss_fn = load_loss_fn(model["loss_fn"])
    reg_fn = load_reg_fn(model["reg_fn"])
    tied_parameters = model["tied_parameters"]
    n_areas = model["n_areas"]

    model_params = {
        "params": params,
        "learning_rates": lrates,
        "EPS": eps,
        "tied_parameters": tied_parameters,
        "population": torch.tensor(population),
        "initial_conditions": initial_conditions,
        "integrator": integrator,
        "n_areas": n_areas,
        "loss_fn": loss_fn,
        "reg_fn": reg_fn,
        "time_step": time_step,
        "momentum_settings": momentum_settings
    }

    if class_ == SpatioTemporalSidarthe.__name__:
        return SpatioTemporalSidarthe(**model_params)
    else:
        raise ValueError("Checkpoint load is supported only for STSidarthe for now")

def load_dataset(dataset: Dict):
    region, data_path = dataset["region"], dataset["data_path"]
    train_size, val_size = dataset["train_size"], dataset["val_size"]

    class_ = dataset["class"]
    if class_ == SirDataModule.__name__:
        return SirDataModule(region, data_path, train_size, val_size)
    elif class_ == SidartheDataModule.__name__:
        return SidartheDataModule(region, data_path, train_size, val_size)
    elif class_ == SpatioTemporalSidartheDataset.__name__:
        return SpatioTemporalSidartheDataset(region, data_path, train_size, val_size)
    else:
        raise ValueError(f"Unknown dataset class {class_}")

def load_integrator(integrator: Dict):
    class_ = integrator["class"]
    t_step = integrator["time_step"]
    if class_ == Euler.__name__:
        return Euler
    elif class_ == Heun.__name__:
        return Heun
    elif class_ == RK4.__name__:
        return RK4
    else:
        raise ValueError(f"Unknown integrator class {class_}")

def load_loss_fn(loss_fn: Union[List, Dict]):

    if isinstance(loss_fn, List):
        return compose_losses([load_loss_fn(lfn) for lfn in loss_fn])

    class_, ignored, weights = loss_fn["class"], loss_fn["ignored"], loss_fn["weights"]

    if class_ == RMSE.__name__:
        return RMSE(weights, ignored)
    elif class_ == NRMSE.__name__:
        return NRMSE(weights, ignored)
    elif class_ == MAE.__name__:
        return MAE(weights, ignored)
    elif class_ == MAPE.__name__:
        return MAPE(weights, ignored)
    else:
        raise ValueError(f"Unknown loss_fn class {class_}")

def load_reg_fn(reg_fn: Dict):
    if isinstance(reg_fn, List):
        return compose_losses([load_reg_fn(rfn) for rfn in reg_fn])

    class_, weight = reg_fn["class"], reg_fn["weight"]

    if class_ == LteZero.__name__:
        return LteZero(weight)
    elif class_ == LogVicinity.__name__:
        return LogVicinity(weight)
    elif class_ == FirstDerivative.__name__:
        time_step = reg_fn["time_step"]
        return FirstDerivative(weight, time_step)
    else:
        raise ValueError(f"Unknown reg_fn class {class_}")

