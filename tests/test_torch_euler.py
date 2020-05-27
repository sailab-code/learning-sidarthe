import pytest
import numpy as np
import torch
from torch_euler import euler, Heun, RK4

# use 64 bit precision to test numerical methods!!!
torch_64 = torch.float64
np_64 = np.float64

# differential equation
def f(T, X):
    return X 

# initial condition
def IC(t):
    return torch.tensor([[1.]], dtype=torch_64)

# integration interval
def integration_interval(steps, ND=1):
    return torch.linspace(0, ND, steps, dtype=torch_64)

# analytical solution
def analytical_solution(t_range):
    return np.exp(t_range, dtype=np_64)

t_range_coarse = integration_interval(steps=10)
solution_coarse = analytical_solution(t_range_coarse)
t_range_medium = integration_interval(steps=100)
solution_medium = analytical_solution(t_range_medium)
t_range_fine = integration_interval(steps=1000)
solution_fine = analytical_solution(t_range_fine)

@pytest.mark.parametrize("method, expected_error, t_range, analytical_solution",
[(euler, torch.tensor(1.3710703675e-01, dtype=torch_64), t_range_coarse, solution_coarse),
 (Heun, torch.tensor(5.1428362153e-03, dtype=torch_64), t_range_coarse, solution_coarse),
 (RK4, torch.tensor(3.1476326630e-06, dtype=torch_64), t_range_coarse, solution_coarse),
 (euler, torch.tensor(1.3602792294e-02, dtype=torch_64), t_range_medium, solution_medium),
 (Heun, torch.tensor(4.5875407627e-05, dtype=torch_64), t_range_medium, solution_medium),
 (RK4, torch.tensor(2.3384139070e-10, dtype=torch_64), t_range_medium, solution_medium),
 (euler, torch.tensor(1.3592542326e-03, dtype=torch_64), t_range_fine, solution_fine),
 (Heun, torch.tensor(4.5361372081e-07, dtype=torch_64), t_range_fine, solution_fine),
 (RK4, torch.tensor(2.3536728122e-14, dtype=torch_64), t_range_fine, solution_fine),
], 
ids=["euler_coarse", "Heun_coarse", "RK4_coarse",
     "euler_medium", "Heun_medium", "RK4_medium", 
     "euler_fine", "Heun_fine", "RK4_fine"])
def test_method(method, expected_error, t_range, analytical_solution):
    numerical_solution = method(f, IC, t_range)
    numerical_solution = numerical_solution.squeeze(1)
    L_inf_err = torch.dist(numerical_solution, analytical_solution, float('inf'))
    assert(torch.isclose(L_inf_err, expected_error))