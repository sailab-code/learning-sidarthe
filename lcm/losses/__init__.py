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

    def __init__(self, weights, ignore_targets=None, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
        self.ignore_targets = ignore_targets if ignore_targets is not None else []

    @abc.abstractmethod
    def target_loss(self, hat, target, mask):
        pass

    def __call__(self, hats, targets, mask, **kwargs):
        losses = {key: self.target_loss(hats[key], targets[key], mask) for key in targets.keys()}

        backward = 0.
        validation = 0.
        for key, loss in losses.items():
            if key in self.ignore_targets:
                continue
            backward = backward + self.weights[key[0]] * loss
            validation = validation + loss

        losses["weighted"] = backward
        losses["unweighted"] = validation

        return losses

    def __str__(self):
        weight_str = ", ".join([f"{key}:{val}" for key, val in self.weights.items()])
        return self.__class__.__name__ + " " + weight_str + f"; ignored: [{','.join(self.ignore_targets)}]"


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

        losses["weighted"] = torch.mean(backward)
        losses["unweighted"] = torch.mean(validation)

        return losses

    def __str__(self):
        return f"{self.__class__.__name__}: {self.weight}"


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

        def __str__(self):
            return "\n".join([lfn.__str__() for lfn in self.loss_fns])

    if name is not None:
        ComposedLoss.__name__ = name
    else:
        ComposedLoss.__name__ = "_".join([
            loss_fn.__class__.__name__
            for loss_fn in loss_fns
        ])

    return ComposedLoss()

