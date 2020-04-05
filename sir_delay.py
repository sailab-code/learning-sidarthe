import numpy as np
import pylab as pl
from ddeint import ddeint

Population = 1
betahat = 0.3
beta = betahat / Population
gamma = 0.05

TS = 1 # daily updating
ND = 200  # number of days of simulation
epsilon = 1e-6  # set the seed of infection
S0 = 1 - epsilon
I0 = epsilon
S0 = S0 * Population
I0 = I0 * Population

INPUT = (S0, I0, 0.0)  # initialization of SIR parameters (Suscettible, Infected, Recovered)

tau = 0 # recovery time (average of 14)

t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)

def omega(t):
    return np.asarray([
      0 if t < 0 else S0,
      0 if t < 0 else I0,
      0
    ])

flag_y = False

def diff_eqs(X, t, tau):
    '''SIR Model'''
    global flag_y
    Y = np.zeros(3)
    V = X(t)

    x_past, y_past, z_past = X(t - tau)


    Y[0] = - beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    Y[2] = gamma * V[1]


    if not flag_y:
        print("entered not flag y")
        if V[1] + Y[1] * TS < 0:
            Y[1] = 0 - V[1]
            flag_y = True
            print("entered true")
    else:
        print("entered else")
        Y[1] = 0
        Y[2] = 0



    return Y

RES = ddeint(diff_eqs, omega, t_range, fargs=(tau,))

# Plotting
a = pl.figure(1)
pl.subplot(211)
pl.plot(t_range, RES[:, 0], '-g', label='Susceptibles')
pl.plot(t_range, RES[:, 2], '-k', label='Recovereds')
pl.legend(loc=0)
pl.title(f"Coronavirus in Italy with tau = {tau}")
pl.xlabel('Time in days')
pl.ylabel('Susceptibles and Recovereds')
pl.plot(t_range, RES[:, 1], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.grid()
a.show()