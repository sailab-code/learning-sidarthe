import torch
from torch.optim.optimizer import Optimizer


class SirOptimizer(Optimizer):
    def __init__(self, params, etas, m, a, b, momentum=True):
        """

        :param params: iterable containing parameters to be optimized
        :param etas: iterable containing learning rates in the same order as the parameter
        :param kwargs: keyword argument required:
            * eta_b, eta_g, eta_d : learning rates for beta, gamma,delta
            * a, b : parameters for learning rate decay
            * alpha: parameter for momentum term
        """

        self.etas = etas
        self.a = a
        self.b = b
        self.m = m
        self.momentum = momentum
        defaults = dict()

        super(SirOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None,
        if closure is not None:
            loss = closure()

        for group in self.param_groups:  # modded learning rates
            for idx, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    continue

                # # print(p.grad)
                # d_p = parameter.grad.data
                # lr_t = -etas[idx] * d_p
                # mu_t = torch.cumprod(torch.flip(torch.cat((torch.ones(1), mu[:-1])), dims=[0]), dim=0)  # 1, mu[0], mu[0]*mu[1], ...
                # update = torch.cumsum(lr_t * mu_t, dim=0)

                d_p = parameter.grad.data
                if self.momentum:
                    times = torch.arange(group["params"][0].shape[0], dtype=torch.float32)
                    # times = times * self.sample_time #added AB
                    mu = torch.sigmoid(self.m * times)
                    eta_mod = self.a / (self.a + self.b * times)
                    etas = torch.tensor(self.etas)
                    etas = etas.unsqueeze(1) * eta_mod.unsqueeze(0)
                    update = [-etas[idx][0] * d_p[0]]
                    for t in range(1, d_p.size(0)):
                        momentum_term = -etas[idx][t] * d_p[t] + mu[t] * update[t - 1]
                        update.append(momentum_term)
                    parameter.data.add_(torch.tensor(update))
                else:
                    parameter.data.add_(-self.etas[idx] * d_p)
