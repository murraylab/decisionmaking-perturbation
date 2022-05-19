'''
Simulation code for Drift Diffusion Model
Author: Norman Lam (normanlam1217@gmail.com)
'''
#import brian_no_units           #Note: speeds up the code
from numpy.fft import rfft,irfft
import time
import numpy as np
from scipy.special import erf
#from scipy.linalg import circulant
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import cProfile
import re
import matplotlib.cm as matplotlib_cm
import math
#import sys
import multiprocessing
import copy


########################################################################################################################
### Initialization
## Flags to run various parts or not

# Parameters.                                                                                                           # Parameters structured in this way: []
dx = 0.008                                                                                                              # grid size
T_dur = 2.                                                                                                              # [s] Duration of simulation
dt = 0.005                                                                                                              # [s] Time-step.


# Parameters consistent with spiking circuit data.                                                                      # Arranged in params in ddm_pdf_genreal as [mu_0,param_mu_x,param_mu_t, sigma_0,param_sigma_x,param_sigma_t, B_0, param_B_t]. Fitted params.
# Mu as drift
mu_0 = 14.3                                                                                                             # [1/s] Constant component of drift rate mu.
# coh_list = np.array([0.0,3.2,6.4,12.8,25.6,51.2])                                                                     # [%] For standard task paradigm and Duration Paradigm.
coh_list = np.array([-51.2, -25.6, -12.8, -6.4, -3.2, 0.0, 3.2, 6.4, 12.8, 25.6, 51.2])                                 # [%] For Pulse Paradigm
mu_0_list = [mu_0*0.01*coh_temp for coh_temp in coh_list]                                                               # [1/s] List of mu_0, to be looped through for tasks.
param_mu_x_OUpos = 6.75                                                                                                 # [1/s] self-coupling parameter, lambda, for elevated E/I circuiut model (reduced EI).
param_mu_x_OUneg = -7.77                                                                                                # [1/s] self-coupling parameter, lambda, for lowered E/I circuiut model (reduced EE).
param_mu_t = 0.                                                                                                         # Parameter for t_dependence of mu.
# Sigma as Noise
sigma_0       = 1.33                                                                                                    # [s^-0.5] Constant component of noise sigma.
param_sigma_x = 0.5                                                                                                     # Parameter for x_dependence of sigma.
param_sigma_t = 0.5                                                                                                     # Parameter for t_dependence of sigma.
# B as Bound
B = 1.                                                                                                                  # Boundary. Assumed to be 1
param_B_t = 1.                                                                                                          # Parameter for t_dependence of B (no x-dep I sps?).


# Declare arrays for usage and storage.
x_list = np.arange(-B, B+0.1*dx, dx)                                                                                    # List of x-grids (Staggered-mesh)
center_matrix_ind  = (len(x_list)-1)/2                                                                                  # index of the center of the matrix. Should be integer by design of x_list
t_list = np.arange(0., T_dur, dt)                                                                                       # t-grids
# pdf_list = np.zeros((len(x_list), len(t_list)))                                                                         # List of probability density functions (pdf)
# pdf_list[center_matrix_ind, 0] = 1.                                                                                     # Initial Condition: All pdf (=1) at center of grid at time 0.


##Pre-defined list of models that can be used, and the corresponding default parameters
# Control model = setting_list[0]
# Elevated E/I model = setting_list[1]
# Lowered E/I model = setting_list[2]
setting_list = [['linear_xt', 'linear_xt', 'constant', 'point_source_center'], ['linear_xt', 'linear_xt', 'constant', 'point_source_center'], ['linear_xt', 'linear_xt', 'constant', 'point_source_center']]     #Define various setting specs.
task_list = ['Fixed_Duration', 'PsychoPhysical_Kernel', 'Duration_Paradigm', 'Pulse_Paradigm']                          # Define various setting specs for each tasks...
task_params_list = [[], [], [0.1*mu_0, T_dur/2.], [0.1*mu_0, T_dur/2.]]
models_list_all = [0,1,2]                                                                                               # List of models to use. See Setting_list
param_mu_x_list = [0., param_mu_x_OUpos, param_mu_x_OUneg]                                                              # List of param_mu (x) input in DDM_pdf_general.
param_mu_t_list = [0., 0., 0.]                                                                                          # List of param_mu (t) input in DDM_pdf_general.
param_sigma_x_list = [0., 0., 0.]                                                                                       # List of param_sigma (x) input in DDM_pdf_general.
param_sigma_t_list = [0., 0., 0.]                                                                                       # List of param_sigma (t) input in DDM_pdf_general.
param_B_t_list = [0., 0., 0.]                                                                                           # List of param_B (t) input in DDM_pdf_general.
labels_list = ['DDM', 'OU+', 'OU-']                                                                                     # Labels for figures
color_list  = ['r', 'g', 'b']                                                                                           # Colors for figures. TEMP: Want r/g/b for DDM/OU+/OU-

########################################################################################################################

# matrix_diffusion_outer = np.zeros((len(x_list), len(x_list)))
# matrix_diffusion_inner = np.zeros((len(x_list), len(x_list)))

matrix_diffusion = np.zeros((len(x_list), len(x_list)))




