import torch

from lcm.losses import TargetLoss

class RMSE(TargetLoss):
    def target_loss(self, hat, target):
        mask = torch.ge(target, 0.)

        return torch.sqrt(
            0.5 * torch.mean(
                torch.pow(target[mask] - hat[mask], 2)
            )
        )

class MAE(TargetLoss):
    def target_loss(self, hat, target):
        mask = torch.ge(target, 0)

        return torch.mean(
            torch.abs(target[mask] - hat[mask])
        )


class MAPE(TargetLoss):
    def target_loss(self, hat, target):
        mask = torch.ge(target, 0)

        return torch.mean(
            torch.abs(
                (target[mask] - hat[mask]) / target[mask]
            )
        )

