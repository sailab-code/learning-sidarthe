from torch import optim
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint
import torch
from matplotlib import pyplot as plt

N = 5
gamma = torch.tensor([[0.3]] * N, requires_grad=True)
beta = torch.tensor([[0.8]] * N, requires_grad=True)
population = 1
epsilon_s = 1e-6
S0 = 1 - epsilon_s
I0 = epsilon_s
ND = 200
TS = 1


class Sir(torch.nn.Module):

    def forward(self, t, y):
        X_t = y
        t = t.long()
        if t < beta.shape[0]:
            beta_t = beta[t] / population
            gamma_t = gamma[t]
        else:
            beta_t = beta[-1] / population
            gamma_t = gamma[-1]

        return torch.cat((
            - beta_t * X_t[0] * X_t[1],
            beta_t * X_t[0] * X_t[1] - gamma_t * X_t[1],
            gamma_t * X_t[1]
        ), dim=0)



epochs = 251
lr = 1e-3
if __name__ == '__main__':
    t_range = torch.arange(0, ND, TS, dtype=torch.float32)
    init_cond = torch.tensor([S0, I0, 0])
    optimizer = optim.SGD([beta, gamma], lr=lr, momentum=0.9)
    for epoch in range(0, epochs):
        print("epoch {}".format(epoch))
        optimizer.zero_grad()
        sol = odeint(Sir(), init_cond, t_range, method='euler')
        # sol = euler(dynamic_f, omega, t_range)
        z_hat = sol[-1][2]

        z_target = torch.tensor([[0.6]])

        loss = torch.pow(z_target - z_hat, 2)
        #print(k)
        loss.backward()
        #print(beta.grad)
        #print(gamma.grad)
        #print(z_hat)
        optimizer.step()
        # update params

        if epoch % 50 == 0:
            a = plt.figure(1)
            plt.plot(t_range.detach().numpy(), sol.detach().numpy())
            plt.grid()
            a.show()

            print("loss: {}".format(loss))
            print("beta: {}".format(beta))
            print("gamma: {}".format(gamma))

    print("loss: {}".format(loss))
    print("beta: {}".format(beta))
    print("gamma: {}".format(gamma))