import numpy as np
import torch
from learning_models.model import LearningModel

b_reg = torch.tensor([1e7])
c_reg = torch.tensor([1e7])
d_reg = torch.tensor([1e7])
bc_reg = torch.tensor([1e9])


class SIR(LearningModel):
    def __init__(self, data, n, x_0, y_0,  configs):
        super(SIR, self).__init__()

        self.x, self.w = data
        self.x = torch.tensor(self.x).view(-1, 1).float()
        self.w = torch.tensor(self.w).float()

        self.configs = configs

        self.n = n
        # pretrained_model definition
        self.b = torch.tensor([0.8], requires_grad=True)
        self.c = torch.tensor([0.25], requires_grad=True)
        self.d = torch.tensor([0.15], requires_grad=True)

        self.x_0, self.y_0 = torch.tensor(x_0, dtype=torch.float), torch.tensor(y_0, dtype=torch.float)

        self.optimizer = self.configs["optimizer"]

    def __call__(self, t):
        print(self.b)
        print(self.c)
        print("Diff" + str(self.b - self.c))
        # print(t)
        b = self.b
        c = self.c
        print(self.d)
        # print("==========")
        time_ratio = (b-c)*t
        exp_bc_ratio = b/(b-c)
        exp_cb_ratio = c/(b-c)

        x_0 = self.x_0
        y_0 = self.y_0
        r = y_0/x_0

        xy_prod1 = (1.0+r).pow(exp_bc_ratio)
        xy_prod2 = (1.0+r*torch.exp(time_ratio)).pow(-exp_bc_ratio)
        x = x_0 * xy_prod1 * xy_prod2  # compute x(t)
        y = y_0 * xy_prod1 * xy_prod2 * torch.exp(time_ratio)  # compute w(t)
        # print(xy_prod1)
        # print(xy_prod2)
        # print(y)

        z_prod1 = (x_0 + y_0).pow(exp_bc_ratio)
        z_prod2 = (x_0 + y_0*torch.exp(time_ratio)).pow(-exp_cb_ratio)
        z = self.n - (z_prod1 * z_prod2)  # compute z_hat(t)

        # w_hat = torch.max(torch.tensor(1e-6), torch.min(self.d, torch.tensor(0.9999))) * z_hat  # d buonded in [0,0.9999]
        w_hat = self.d * z  # d buonded in [0,0.9999]

        return w_hat, z, x, y

    def fit(self, params):
        # self.x_0.data.fill_(params["initial_x"])
        # self.y_0.data.fill_(params["initial_y"])

        optimizer = self.optimizer([{'params': self.b, 'lr': params["lrb"]},
                                    {'params': self.c, 'lr': params["lrc"]},
                                    {'params': self.d, 'lr': params["lrd"]}])

        risk = 1e12
        n_epochs = self.configs["n_epochs"]
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            w_hat, _, _, _ = self(self.x)
            mse_loss = self.loss(self.w, w_hat)
            b = torch.abs(self.b)
            c = torch.abs(self.c)
            d = torch.abs(self.d)
            loss = mse_loss + (b.ge(1.0) * b * b_reg) + (c.ge(1.0) * c * c_reg) + (d.ge(1.0) * d * d_reg) + \
                   (self.b.le(0.0) * b * b_reg) + (self.c.le(0.0) * c * c_reg) + (d.le(0.0) * d * d_reg) +\
                   (self.b - self.c).le((self.b - self.c)/10)*bc_reg

            # if int(torch.sum((torch.isinf(loss) + torch.isnan(loss)).int()).detach().numpy()) == 1:
            #     print("explode")
            #     mse_loss = torch.tensor(1e12)
            # else:
            loss.backward()
            optimizer.step()
            risk = np.sqrt(2 * mse_loss.detach().numpy())  #
            print(risk)

        return risk

    @staticmethod
    def loss(y, y_hat):
        return torch.mean(0.5 * (y_hat.view(-1) - y) * (y_hat.view(-1) - y))  # MSELoss

    def eval(self, x, w):
        w_hat, z, x, y = self(x)
        return np.sqrt(2 * self.loss(w, w_hat).detach().numpy()), z, x, y

