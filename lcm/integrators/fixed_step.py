import torch

from lcm.integrators import Integrator


class Euler(Integrator):
    """
    Euler integration method.
    """

    def __call__(self, diff_equations, initial_conditions, time_grid):
        values = initial_conditions

        for i in range(0, time_grid.shape[0] - 1):
            t_i = time_grid[i]
            t_next = time_grid[i + 1]
            y_i = values[i]
            dt = t_next - t_i
            dy = diff_equations(t_i, y_i) * dt
            y_next = y_i + dy
            y_next = y_next.unsqueeze(0)
            values = torch.cat((values, y_next), dim=0)

        return values



class Heun(Integrator):
    """
    Heun integration method.
    """

    def __call__(self, diff_equations, initial_conditions, time_grid):
        values = initial_conditions

        for i in range(0, time_grid.shape[0] - 1):
            t_i = time_grid[i]
            t_next = time_grid[i + 1]
            y_i = values[i]
            dt = t_next - t_i
            dt = dt.unsqueeze(1)
            f1 = diff_equations(t_i, y_i)
            f2 = diff_equations(t_next, y_i + dt * f1)
            dy = 0.5 * dt * (f1 + f2)
            y_next = y_i + dy
            y_next = y_next.unsqueeze(0)
            values = torch.cat((values, y_next), dim=0)

        return values


class RK4(Integrator):
    """
    RK4 integration method.
    """

    def __call__(self, diff_equations, initial_conditions, time_grid):
        values = initial_conditions

        for i in range(0, time_grid.shape[0] - 1):
            t_i = time_grid[i]
            t_next = time_grid[i + 1]
            y_i = values[i]
            dt = t_next - t_i
            dtd2 = 0.5 * dt
            f1 = diff_equations(t_i, y_i)
            f2 = diff_equations(t_i + dtd2, y_i + dtd2 * f1)
            f3 = diff_equations(t_i + dtd2, y_i + dtd2 * f2)
            f4 = diff_equations(t_next, y_i + dt * f3)
            dy = 1 / 6 * dt * (f1 + 2 * (f2 + f3) + f4)
            y_next = y_i + dy
            y_next = y_next.unsqueeze(0)
            values = torch.cat((values, y_next), dim=0)

        return values