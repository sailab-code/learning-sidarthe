import torch
import torch.nn as nn
from learning_models.model import LearningModel
import numpy as np


class Logistic(LearningModel):
    def __init__(self, data, configs):
        super(Logistic, self).__init__()
        self.x, self.y = data
        self.x = torch.tensor(self.x).view(-1, 1).float()
        self.y = torch.tensor(self.y).float()

        self.configs = configs

        # model definition
        self.wx_b = nn.Linear(1, 1)
        # nn.init.xavier_normal_(wx_b.weight)
        # nn.init.uniform_(m, a=1.0, b=max(y))
        self.m = torch.tensor([0.0], requires_grad=True)

        self.optimizer = self.configs["optimizer"]

    def __call__(self, x):
        return self.m * torch.sigmoid(self.wx_b(x))

    def fit(self, params):

        # variables initialization
        self.wx_b.weight.data.fill_(params["initial_w"])
        self.wx_b.bias.data.fill_(params["initial_b"])

        self.m.data.fill_(params["initial_m"])

        optimizer = self.optimizer([{'params': self.wx_b.weight, 'lr': params["lrw"]},
                                    {'params': self.wx_b.bias, 'lr': params["lrb"]},
                                    {'params': self.m, 'lr': params["lrm"]}])

        max_no_improve = 15
        thresh = 0.5
        last_improve, best = 0, 1e12
        n_epochs = self.configs["n_epochs"]
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            y_hat = self(self.x)
            loss = self.loss(self.y, y_hat)
            loss.backward()
            optimizer.step()
            risk = np.sqrt(2*loss.detach().numpy())  # risk
            last_improve = 0 if risk < best + thresh else last_improve + 1
            best = risk if risk < best else best

            if last_improve >= max_no_improve:
                print("Early Stop at epoch: %d" % epoch)
                break

        return best

    @staticmethod
    def loss(y, y_hat):
        return torch.mean(0.5 * (y_hat.view(-1) - y)*(y_hat.view(-1) - y))  # MSELoss

    def eval(self, x):
        return self(x)
