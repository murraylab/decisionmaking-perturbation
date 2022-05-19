'''
Parameter recovery code for Drift Diffusion Model simulation
It is recommended to run this on computer-cluster resources, speeding up with parallelization.
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
import cProfile
import re
import datetime

########################################################################################################################
### Initialization
## Flags to run various parts or not
Flag_Pulse = 1

#Load parameters and functions
# execfile('DDM_parameters.py')                                                                                           # python 2
# execfile('DDM_functions.py')                                                                                            # python 2
# execfile('DDM_sim_pulse_param_recovery.py')                                                                             # python 2
exec(open("DDM_parameters.py").read())                                                                                  # python 3
exec(open("DDM_functions.py").read())                                                                                   # python 3
exec(open("DDM_sim_pulse_param_recovery.py").read())                                                                    # python 3






########################################################################################################################


### Psychophysical Kernel/ Duration Paradigm/ Pulse Paradigm...
# Pulse Paradigm...

n_trials = 1000 # number of trials for each condition (11 coh condi, x 20 pulse condi)

if Flag_Pulse:
    t_onset_list_pulse = np.arange(0., T_dur, 0.1)
    models_list = [1]                                   # List of models to use. See Setting_list. 1 or 2.
    param_mu_x_list[1] = 5                              # N.B.: this is the param (lambda) to vary in each simulation, from -10 to 10 (stepsize 1).

    Prob_final_corr_pulse  = np.zeros((len(t_onset_list_pulse), len(mu_0_list), len(models_list)))


    ## For each models, simulate the probability to be correct/erred/undec for various mu and t_onset_pulse
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        for i_mu0 in range(len(mu_0_list)):
            mu_2use = mu_0_list[i_mu0]
            for i_ton in range(len(t_onset_list_pulse)):
                t_onset_temp = t_onset_list_pulse[i_ton]

                for i_sim in range(n_trials):
                    x_sim_1 = sim_DDM_pulse(mu_2use,'OU',param_mu_x_list[index_model_2use], sigma_0,'constant', B,'constant', t_onset_temp)   # x_sim_1 = 1 if corr, 0 if error, 0.5 if undecided.
                    Prob_final_corr_pulse[i_ton, i_mu0, i_models] += x_sim_1 / float(n_trials)


        ## For each model and each t_onset_pulse, fit the psychometric function
        psychometric_params_list_pulse = np.zeros((3 , len(t_onset_list_pulse), len(models_list)))
        param_fit_0_pulse = [2. ,1., 0.]                                                                                # Initial guess for param_pm for Psychometric_fit_P.
        for i_models in range(len(models_list)):
            index_model_2use = models_list[i_models]
            for i_ton in range(len(t_onset_list_pulse)):
                t_onset_temp = t_onset_list_pulse[i_ton]
                res_temp = minimize(Psychometric_fit_P, param_fit_0_pulse, args = ([Prob_final_corr_pulse[i_ton,:,i_models]]))     # Note that mu_0_list is intrinsically defined in the Psychometric_fit_P function
                psychometric_params_list_pulse[:,i_ton,i_models] = res_temp.x




    #### For visualizing, block for QL
    figP = plt.figure(figsize=(8,10.5))
    axP1 = figP.add_subplot(411)
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axP1.plot(t_onset_list_pulse, psychometric_params_list_pulse[2,:,i_models]*100./mu_0, color=color_list[index_model_2use], label=labels_list[index_model_2use] )               # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    axP1.set_ylabel('Shift')
    axP1.set_title('Psychometric function Shift')
    axP1.legend(loc=1)

    axP2 = figP.add_subplot(412)
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axP2.plot(t_onset_list_pulse, psychometric_params_list_pulse[0,:,i_models]*100./mu_0, color=color_list[index_model_2use], label=labels_list[index_model_2use] )               # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    axP2.set_ylabel('Threshold')
    axP2.set_title('Psychometric function Threshold')
    axP2.legend(loc=1)

    axP3 = figP.add_subplot(413)
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axP3.plot(t_onset_list_pulse, psychometric_params_list_pulse[1,:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    axP3.set_ylabel('Order')
    axP3.set_title('Psychometric function Slope/Order')
    axP2.legend(loc=4)

    axP4 = figP.add_subplot(414)
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axP4.plot(mu_0_list, Prob_final_corr_pulse[-1,:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use], linestyle="--" )
        axP4.plot(mu_0_list, 0.5 + np.sign(mu_0_list)*0.5*(1. - np.exp(-(np.sign(mu_0_list)*(mu_0_list+psychometric_params_list_pulse[2,-1,i_models])/psychometric_params_list_pulse[0,-1,i_models]) **psychometric_params_list_pulse[1,-1,i_models])) , color=color_list[index_model_2use], label=labels_list[index_model_2use]+'_long' , linestyle=":" )
    axP4.set_ylabel('Correct Probability')
    axP4.set_title('Correct Probability')
    axP4.legend(loc=4)

    figP.savefig('Pulse_paradigm_for_p_recovery.pdf')



    ############## Now try refit model
    mu_sigma_lambda_fitted_recov_list_fit_all = np.zeros((3, len(models_list)))

    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]

        ### Fit mu/sigma/lambda simultaneously, over Pcorr as a function of coherence and i_ton.
        mu_sigma_lambda_init = [10, 1., 0.]
        res_temp = minimize(fit_Pulse_param_recov_fit_all, mu_sigma_lambda_init, args=(Prob_final_corr_pulse[:,:,i_models]))
        mu_sigma_lambda_fitted_recov_list_fit_all[:,i_models] = res_temp.x


    np.save("pulse_p_recovery_lambda_for_sim.npy", param_mu_x_list[3])                                                  # lambda used in the simulation.
    np.save("pulse_p_recovery_params_refitted.npy", mu_sigma_lambda_fitted_recov_list_fit_all)                          # Recovered parameters of mu/sigma/lambda.



