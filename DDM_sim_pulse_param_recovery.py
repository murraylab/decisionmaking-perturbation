'''
Simulation code for Drift Diffusion Model, parameter recovery
Author: Norman Lam (normanlam1217@gmail.com)
'''
#import brian_no_units           #Note: speeds up the code
from numpy.fft import rfft,irfft
import time
import numpy as np
from scipy.special import erf
#from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cProfile
import re



########################################################################################################################
### Initialization: Parameters
dt_sim = 0.001                  # [s] % simulation run time
tau_bound = 1.                  # [s] % For collapisn bounds. Not used.


########################################################################################################################
### Functions
# Function simulating one trial of the time-varying drift/bound DDM
def sim_DDM_pulse(mu, f_mu_setting, param_mu_x_temp, sigma, f_sigma_setting, B, f_bound_setting, param_pulse_task, seed=1):
    '''
    f_mu, f_sigma, f_Bound are callable functions
    '''
    # Set random seed
    # np.random.seed(seed)

    # Initialization
    x = 0
    t = 0

    # Storage
    # x_plot = [x]
    # t_plot = [t]

    # Looping through time
    while abs(x)<=f_bound(B, x, t, tau_bound, f_bound_setting):
        x += (f_mu(mu, x, t, f_mu_setting, param_mu_x_temp) + f_mu1_task(t, task_list[3], param_pulse_task))*dt_sim + np.random.randn()*f_sigma(sigma, x, t, f_sigma_setting)*np.sqrt(dt_sim)
        t += dt_sim

        # x_plot.append(x)
        # t_plot.append(t)

        if x > f_bound(B, x, t, tau_bound, f_bound_setting):
            rt = t
            choice = 1. # choose left.
            break
        if x < -f_bound(B, x, t, tau_bound, f_bound_setting):
            rt = t
            choice = 0. # choose right.
            break
        if t>2.:                                                                                                        # Block this if loop to become Reaction time task.
            rt = 0
            choice = 0.5 # undecided, choice undefined originally
            break


    return choice
    # return choice, rt, x_plot, t_plot                                                                                 # If want to output more varaibles.





## Effect of drift=mu to particle
def f_mu(mu, x, t, f_mu_setting, param_mu_x_temp):
    if f_mu_setting == 'constant':
        return mu
    if f_mu_setting == 'OU':
        a = param_mu_x_temp
        return mu + a*x
    if f_mu_setting == 't_term':
        b = 2.*mu
        return mu + b*t
    # Add more if needed.

## Effect of noise=sigma to particle
def f_sigma(sigma, x, t, f_sigma_setting):
    if f_sigma_setting == 'constant':
        return sigma
    # N.B.: Below not used. If use need to add a fitting param similar to f_mu.
    if f_sigma_setting == 'OU':
        a = 0.5
        return sigma + a*x
    if f_sigma_setting == 't_term':
        b = 0.5
        return sigma + b*t
    # Add more if needed.

## Bound at time t.
def f_bound(B, x, t, tau, f_bound_setting):
    if f_bound_setting == 'constant':
        return B
    if f_bound_setting == 'collapsing_linear':
        return B - t/tau
    if f_bound_setting == 'collapsing_exponential':
        return B*np.exp(-t/tau)
    # Add more if needed.


