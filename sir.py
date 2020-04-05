import scipy.integrate as spi
import numpy as np
import pylab as pl

#
# parameters with R0=2.4 - beginning of coronavirus
# SIR parameters

Population = 1
betahat = 0.8
beta = betahat / Population
gamma = 0.3

#
TS = 1  # daily updating
ND = 100  # mumber of days of simulation
epsilon = 1e-6  # set the seed of infection
S0 = 1 - epsilon
I0 = epsilon
S0 = S0 * Population
I0 = I0 * Population

INPUT = (S0, I0, 0.0)  # initialization of SIR parameters (Suscettible, Infected, Recovered)

tau = int(10)  # recovery time (average of 14)

i_history = np.zeros(2 * ND + 1)  # history of y(t) values

flag_y = False
t_change_flag = 0
t_last = 0
t_second_last = 0


def diff_eqs_2(INP, t):
    '''SIR Model'''
    global i_history
    global flag_y
    global t_change_flag
    global t_last
    global t_second_last
    Y = np.zeros(3)
    V = INP
    t_past = int(t - tau)  # t minus tau
    i_history[int(t)] = V[1]

    if t_past >= 0:
        i_past = i_history[t_past]
    else:
        i_past = 0

    Y[0] = - beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * i_past
    Y[2] = gamma * i_past

    if t_change_flag >= t:
        flag_y = False

    dt = t - t_second_last
    if not flag_y:
        print("entered not flag y")
        if V[1] + Y[1] * dt < 0 and dt > 0:
            Y[1] = (-V[1]) / dt
            flag_y = True
            t_change_flag = t
            print("entered true")
    else:
        print("entered else")
        Y[1] = 0
        Y[2] = 0

    print(f"t: {t}, \n"
          f"t-tau: {t - tau}\n"
          f"x: {V[0]}\n"
          f"x_dot: {Y[0]}\n"
          f"y: {V[1]}\n"
          f"y_dot: {Y[1]}\n"
          f"z: {V[2]}\n"
          f"z_dot: {Y[2]}\n"
          f"dt: {dt}\n"
          f"t_last: {t_last}\n"
          f"t_second_last: {t_second_last}\n\n")

    if t != t_last:
        t_second_last = t

    t_last = t
    return Y  # For odeint


def diff_eqs(INP, t):
    '''SIR Model'''
    Y = np.zeros((3))
    V = INP

    Y[0] = - beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    Y[2] = gamma * V[1]
    return Y  # For odeint


t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)

RES = spi.odeint(diff_eqs, INPUT, t_range)

# Ploting
a = pl.figure(1)
pl.subplot(211)
pl.plot(RES[:, 0], '-g', label='Susceptibles')
pl.plot(RES[:, 2], '-k', label='Recovereds')
pl.legend(loc=0)
pl.title('Coronavirus in Italy')
pl.xlabel('Time in days')
pl.ylabel('Susceptibles and Recovereds')
pl.subplot(212)
pl.plot(RES[:, 1], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
a.show()

RES2 = spi.odeint(diff_eqs_2, INPUT, t_range)

# Ploting
b = pl.figure(2)
pl.plot(RES2[:, 0], '-g', label='Susceptibles')
pl.plot(RES2[:, 2], '-k', label='Recovereds')
pl.legend(loc=0)
pl.title('Coronavirus in Italy (with tau = ' + str(tau) + ')')
pl.xlabel('Time in days')
pl.ylabel('Susceptibles and Recovereds')
pl.plot(RES2[:, 1], '-r', label='Infectious')
pl.xlabel('Time')
pl.ylabel('Infectious')
pl.grid()
b.show()
