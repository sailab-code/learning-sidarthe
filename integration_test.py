import torch

from torch_euler import Heun, euler
from torch.optim import Adam
from matplotlib import pyplot as plt

beta = torch.tensor(1., requires_grad=True)

def f(t,x):
    return beta * x

def f2(t,x):
    return 3. * x

def omega(t):
    return torch.tensor([[1.]])

n_epochs = 100000
t_grid = torch.linspace(0, 3, 100)

target = torch.exp(3*t_grid)
target_2 = Heun(f2, omega, t_grid)

fig = plt.figure()
plt.plot(t_grid, target, color="r")
plt.plot(t_grid, target_2, color="b")
plt.show()
# exit()

optimizer = Adam([beta])
for epoch in range(1, n_epochs+1):
    sol = Heun(f, omega, t_grid)
    loss = torch.mean(0.5 * torch.abs(sol[:,0] - target))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        fig = plt.figure()
        plt.plot(t_grid, sol[:,0].detach().numpy())
        plt.title(f"$\\beta$ == {beta}")
        fig.show()
        print(beta)
        print(loss)




