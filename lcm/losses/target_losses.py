import torch

from lcm.losses import TargetLoss


class RMSE(TargetLoss):
    """
    Root Mean Square Error loss.
    """

    def target_loss(self, hat, target, mask):
        """
        Computes the RMSE loss on target values
        greater or equal than 0.

        :param hat: A tensor of predictions.
        :param target: A tensor with targets.
        :param mask: An tensor for masking elements
        :return: A scalar value (still tensor) corresponding
        to the RMSE between inferences and targets.
        Values of the target lower or equal than 0 are NOT
        accounted in the computation.
        """

        missing_data_mask = torch.ge(target, 0.)
        mask = mask * missing_data_mask
        mask = mask.squeeze()

        return torch.sqrt(
            0.5 * torch.sum(torch.pow(target[mask] - hat[mask], 2))/torch.sum(mask)
        )


class NRMSE(RMSE):
    """

    """
    def __call__(self, hats, targets, mask, **kwargs):
        losses = {key: self.target_loss(hats[key], targets[key], mask) for key in targets.keys()}

        # normalization wrt compartment values
        norm_weights = {key: torch.mean(targets[key]) for key in targets.keys()}
        max_average = torch.max(torch.tensor([value for value in norm_weights.values()]))
        norm_weights = {
            key: max_average / avg if avg > 0.0 else 0.0 for key, avg in norm_weights.items()
        }

        backward = 0.
        validation = 0.
        for key, loss in losses.items():
            backward = backward + norm_weights[key] * self.weights[key[0]] * loss
            validation = validation + loss

        losses["weighted"] = backward
        losses["unweighted"] = validation

        return losses

class MAE(TargetLoss):
    """
    Mean Absolute Error.
    """

    def target_loss(self, hat, target, mask):
        """
        Computes the MAE loss on positive targets.

        :param hat: A tensor of predictions.
        :param target: A tensor with targets.
        :param mask: An tensor for masking elements
        :return: A scalar value (still tensor) corresponding
        to the MAE between inferences and targets.
        Values of the target lower or equal than 0 are NOT
        accounted in the computation.
        """

        missing_data_mask = torch.ge(target, 0.)
        mask = mask * missing_data_mask
        mask = mask.squeeze()

        return torch.sum(torch.abs(target[mask] - hat[mask]))/torch.sum(mask)


class MAPE(TargetLoss):
    """
    Mean Absolute Percentage Error.
    """

    def target_loss(self, hat, target, mask):
        """
        Computes the MAPE loss on positive targets.

        :param hat: A tensor of predictions.
        :param target: A tensor with targets.
        :param mask: An tensor for masking elements
        :return: A scalar value (still tensor) corresponding
        to the MAPE between inferences and targets.
        Values of the target lower or equal than 0 are NOT
        accounted in the computation.
        """

        missing_data_mask = torch.ge(target, 0.)
        mask = mask * missing_data_mask
        mask = mask.squeeze()

        return torch.sum(torch.abs((target[mask] - hat[mask]) / target[mask]))/torch.sum(mask)

