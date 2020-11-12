import abc
import torch


class Loss(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass


class TargetLoss(Loss):
    type = 'target'

    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    @abc.abstractmethod
    def target_loss(self, hat, target):
        pass

    def __call__(self, hats, targets, **kwargs):
        losses = {key: self.target_loss(hats[key], targets[key]) for key in targets.keys()}

        backward = 0.
        validation = 0.
        for key, loss in losses.items():
            backward = backward + self.weights[key[0]] * loss
            validation = validation + loss

        losses["backward"] = backward
        losses["validation"] = validation

        return losses

    def __str__(self):
        weight_str = ", ".join([f"{key}:{val}" for key, val in self.weights.items()])
        return self.__class__.__name__ + " " + weight_str


class RegularizationLoss(Loss):
    type = 'regularization'

    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    @abc.abstractmethod
    def param_loss(self, param):
        pass

    def __call__(self, params):
        losses = {key: self.param_loss(param) for key, param in params.items()}

        backward = 0.
        validation = 0.

        for key, loss in losses.items():
            backward = backward + self.weight * loss
            validation = validation + loss

        losses["backward"] = torch.mean(backward)
        losses["validation"] = torch.mean(validation)

        return losses


def compose_losses(loss_fns, name=None):
    class ComposedLoss(Loss):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            if len({loss_fn.type for loss_fn in loss_fns}) != 1:
                raise ValueError("You can compose only losses of the same type")
            self.loss_fns = loss_fns

        def __call__(self, *args):
            total_losses = {}

            for loss_fn in self.loss_fns:
                loss = loss_fn(*args)
                total_losses = {
                    key: total_losses.get(key, 0.) + value
                    for key, value in loss.items()
                }

            return total_losses

    if name is not None:
        ComposedLoss.__name__ = name
    else:
        ComposedLoss.__name__ = "_".join([
            loss_fn.__class__.__name__
            for loss_fn in loss_fns
        ])

    return ComposedLoss()
