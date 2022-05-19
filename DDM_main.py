'''
Simulation code for Drift Diffusion Model
Note that this code precedes the pyddm package (https://pyddm.readthedocs.io/en/stable/), which is potentially more convenient to use.
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
## Flags to run various tasks or not
Flag_Pulse = 1
Flag_Duration = 1
Flag_PK = 0

#Load parameters and functions
# execfile('DDM_parameters.py')                                                                                         # python 2
# execfile('DDM_functions.py')                                                                                          # python 2
exec(open("DDM_parameters.py").read())                                                                                  # python 3
exec(open("DDM_functions.py").read())                                                                                   # python 3


########################################################################################################################
## Vary coherence and do each tasks (fixed-time, psychophysical kernel, Duration Paradigm, Pulse Paradigm)
models_list = [0,1,2]                                #List of models to use. See Setting_list

Prob_final_corr  = np.zeros((len(mu_0_list), len(models_list)))                                                         # Array to store the total correct probability for each mu & model.
Prob_final_err   = np.zeros((len(mu_0_list), len(models_list)))                                                         # Array to store the total erred probability for each mu & model.
Prob_final_undec = np.zeros((len(mu_0_list), len(models_list)))                                                         # Array to store the total undecided probability for each mu & model.
Mean_Dec_Time    = np.zeros((len(mu_0_list), len(models_list)))                                                         # Array to store the total correct probability for each mu & model.

Prob_final_corr_Analy  = np.zeros((len(mu_0_list), 2))
Prob_final_err_Analy   = np.zeros((len(mu_0_list), 2))
Prob_final_undec_Analy = np.zeros((len(mu_0_list), 2))
Mean_Dec_Time_Analy    = np.zeros((len(mu_0_list), 2))


## Define an array to hold the data of simulations over all models, using the original parameters (mu_0)
Prob_final_corr_0  = np.zeros((len(models_list)))
Prob_final_err_0   = np.zeros((len(models_list)))
Prob_final_undec_0 = np.zeros((len(models_list)))
Mean_Dec_Time_0    = np.zeros((len(models_list)))

traj_mean_pos_all = np.zeros((len(t_list), len(models_list)))

### Compute the probability distribution functions for the correct and erred choices
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    for i_mu0 in range(len(mu_0_list)):
        mu_temp = mu_0_list[i_mu0]
        (Prob_list_corr_temp, Prob_list_err_temp) = DDM_pdf_general([mu_temp, param_mu_x_list[index_model_2use], param_mu_t_list[index_model_2use], sigma_0, param_sigma_x_list[index_model_2use], param_sigma_t_list[index_model_2use], B, param_B_t_list[index_model_2use]], index_model_2use, 0)
        Prob_list_sum_corr_temp  = np.sum(Prob_list_corr_temp)
        Prob_list_sum_err_temp  = np.sum(Prob_list_err_temp)
        Prob_list_sum_undec_temp  = 1 - Prob_list_sum_corr_temp - Prob_list_sum_err_temp


        #Outputs...
        # Forced Choices: The monkey will always make a decision: Split the undecided probability half-half for corr/err choices.
        Prob_final_undec[i_mu0, i_models] = Prob_list_sum_undec_temp
        Prob_final_corr[i_mu0, i_models]  = Prob_list_sum_corr_temp + Prob_final_undec[i_mu0, i_models]/2.
        Prob_final_err[i_mu0, i_models]   = Prob_list_sum_err_temp  + Prob_final_undec[i_mu0, i_models]/2.
        # Mean_Dec_Time[i_mu0, i_models]    = np.sum((Prob_list_corr_temp+Prob_list_err_temp) *t_list) / np.sum((Prob_list_corr_temp+Prob_list_err_temp))   # Regardless of choice made. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.
        Mean_Dec_Time[i_mu0, i_models]    = np.sum((Prob_list_corr_temp)*t_list) / np.sum((Prob_list_corr_temp))        # Only correct choice. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.

        ##Normalize to fit to the analytical solution. (Anderson 1960)
        if index_model_2use ==1 or index_model_2use ==2:
            Prob_final_corr[i_mu0, i_models] = Prob_list_sum_corr_temp / (Prob_list_sum_corr_temp + Prob_list_sum_err_temp)
            Prob_final_err[i_mu0, i_models]  = Prob_list_sum_err_temp / (Prob_list_sum_corr_temp + Prob_list_sum_err_temp)

        ## Analytical solutions (normalized) for PyDDM and CB_Linear, computed if they are in model_list
        if index_model_2use ==0 or index_model_2use==1:
            (Prob_list_corr_Analy_temp, Prob_list_err_Analy_temp) = DDM_pdf_analytical([mu_temp, sigma_0, param_mu_x_list[index_model_2use], B, -param_B_t_list[index_model_2use]], index_model_2use, 0)                               # PyDDM
            Prob_list_sum_corr_Analy_temp  = np.sum(Prob_list_corr_Analy_temp)
            Prob_list_sum_err_Analy_temp   = np.sum(Prob_list_err_Analy_temp)
            Prob_list_sum_undec_Analy_temp = 1 - Prob_list_sum_corr_Analy_temp - Prob_list_sum_err_Analy_temp
            #Output
            # Forced Choices: The monkey will always make a decision: Split the undecided probability half-half for corr/err choices. Actually don't think the analytical solution has undecided trials...
            Prob_final_undec_Analy[i_mu0, i_models] = Prob_list_sum_undec_Analy_temp
            Prob_final_corr_Analy[i_mu0, i_models]  = Prob_list_sum_corr_Analy_temp + Prob_final_undec_Analy[i_mu0, i_models]/2.
            Prob_final_err_Analy[i_mu0, i_models]   = Prob_list_sum_err_Analy_temp  + Prob_final_undec_Analy[i_mu0, i_models]/2.
            # Mean_Dec_Time_Analy[i_mu0, i_models]    = np.sum((Prob_list_corr_Analy_temp+Prob_list_err_Analy_temp) *t_list) / np.sum((Prob_list_corr_Analy_temp+Prob_list_err_Analy_temp))      # Regardless of choices. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.
            Mean_Dec_Time_Analy[i_mu0, i_models]    = np.sum((Prob_list_corr_Analy_temp)*t_list) / np.sum((Prob_list_corr_Analy_temp))                                                           # Only consider correct choices. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.

    ## Compute the default models (based on spiking circuit) for the various models.
    (Prob_list_corr_0_temp, Prob_list_err_0_temp) = DDM_pdf_general([mu_0_list[0], param_mu_x_list[index_model_2use], param_mu_t_list[index_model_2use], sigma_0, param_sigma_x_list[index_model_2use], param_sigma_t_list[index_model_2use], B, param_B_t_list[index_model_2use]], index_model_2use, 0)
    Prob_list_sum_corr_0_temp  = np.sum(Prob_list_corr_0_temp)
    Prob_list_sum_err_0_temp   = np.sum(Prob_list_err_0_temp)
    Prob_list_sum_undec_0_temp = 1. - Prob_list_sum_corr_0_temp - Prob_list_sum_err_0_temp
    #Output
    Prob_final_corr_0[i_models]  = Prob_list_sum_corr_0_temp
    Prob_final_err_0[i_models]   = Prob_list_sum_err_0_temp
    Prob_final_undec_0[i_models] = Prob_list_sum_undec_0_temp
    Mean_Dec_Time_0[i_models]    = np.sum((Prob_list_corr_0_temp+Prob_list_err_0_temp) *t_list) / np.sum((Prob_list_corr_0_temp+Prob_list_err_0_temp))


### Plot correct probability, erred probability, indecision probability, and mean decision time.
fig1 = plt.figure(figsize=(8,10.5))
ax11 = fig1.add_subplot(411)
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    ax11.plot(coh_list, Prob_final_corr[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    if index_model_2use ==0 or index_model_2use==1:
        ax11.plot(coh_list, Prob_final_corr_Analy[:,i_models], color=color_list[index_model_2use], linestyle=':')       #, label=labels_list[index_model_2use]+"_A" )
ax11.set_ylabel('Probability')
ax11.set_title('Correct Probability')
ax11.legend(loc=4)

ax12 = fig1.add_subplot(412)
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    ax12.plot(coh_list, Prob_final_err[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    if index_model_2use ==0 or index_model_2use==1:
        ax12.plot(coh_list, Prob_final_err_Analy[:,i_models], color=color_list[index_model_2use], linestyle=':')        #, label=labels_list[index_model_2use]+"_A" )

ax12.set_ylabel('Probability')
ax12.set_title('Erred Probability')
ax12.legend(loc=1)


ax13 = fig1.add_subplot(413)
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    ax13.plot(coh_list, Prob_final_undec[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    if index_model_2use ==0 or index_model_2use==1:
        ax13.plot(coh_list, Prob_final_undec_Analy[:,i_models], color=color_list[index_model_2use], linestyle=':')      #, label=labels_list[index_model_2use]+"_A" )
ax13.set_ylabel('Probability')
ax13.set_title('Indecision Probability')
ax13.legend(loc=1)

ax14 = fig1.add_subplot(414)
for i_models in range(len(models_list)):
    index_model_2use = models_list[i_models]
    ax14.plot(coh_list, Mean_Dec_Time[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    if index_model_2use ==0 or index_model_2use==1:
        ax14.plot(coh_list, Mean_Dec_Time_Analy[:,i_models], color=color_list[index_model_2use], linestyle=':')         #, label=labels_list[index_model_2use]+"_A" )
ax14.set_xlabel('Coherence (%)')
ax14.set_ylabel('Time (s)')
ax14.set_title('Mean Decision Time')
ax14.legend(loc=3)

fig1.savefig('Fixed_Task_Performance.pdf')

########################################################################################################################
### Task Paradigms

# Pulse Paradigm
if Flag_Pulse:
    t_onset_list_pulse = np.arange(0., T_dur, 0.1)
    models_list = [0,1,2]                                #List of models to use. See Setting_list

    Prob_final_corr_pulse  = np.zeros((len(t_onset_list_pulse), len(mu_0_list), len(models_list)))
    Prob_final_err_pulse   = np.zeros((len(t_onset_list_pulse), len(mu_0_list), len(models_list)))
    Prob_final_undec_pulse = np.zeros((len(t_onset_list_pulse), len(mu_0_list), len(models_list)))
    Mean_Dec_Time_pulse    = np.zeros((len(t_onset_list_pulse), len(mu_0_list), len(models_list)))


    ## For each models, find the probability to be correct/erred/undec for various mu and t_onset_pulse
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        for i_mu0 in range(len(mu_0_list)):
            mu_2use = mu_0_list[i_mu0]
            for i_ton in range(len(t_onset_list_pulse)):
                t_onset_temp = t_onset_list_pulse[i_ton]
                (Prob_list_corr_pulse_temp, Prob_list_err_pulse_temp) = DDM_pdf_general([mu_2use, param_mu_x_list[index_model_2use], param_mu_t_list[index_model_2use], sigma_0, param_sigma_x_list[index_model_2use], param_sigma_t_list[index_model_2use], B, param_B_t_list[index_model_2use], t_onset_temp], index_model_2use, 3)                               # PyDDM
                Prob_list_sum_corr_pulse_temp = np.sum(Prob_list_corr_pulse_temp)
                Prob_list_sum_err_pulse_temp  = np.sum(Prob_list_err_pulse_temp)

                Prob_list_sum_undec_pulse_temp  = 1. - Prob_list_sum_corr_pulse_temp - Prob_list_sum_err_pulse_temp

                #Outputs...
                Prob_final_undec_pulse[i_ton, i_mu0, i_models] = Prob_list_sum_undec_pulse_temp
                Prob_final_corr_pulse[i_ton, i_mu0, i_models]  = Prob_list_sum_corr_pulse_temp + Prob_final_undec_pulse[i_ton, i_mu0, i_models]/2.
                Prob_final_err_pulse[i_ton, i_mu0, i_models]   = Prob_list_sum_err_pulse_temp  + Prob_final_undec_pulse[i_ton, i_mu0, i_models]/2.
                # Mean_Dec_Time_pulse[i_ton, i_mu0, i_models]    = np.sum((Prob_list_corr_pulse_temp+Prob_list_err_pulse_temp) *t_list) / np.sum((Prob_list_corr_pulse_temp+Prob_list_err_pulse_temp))        # Regardless of choices. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.
                Mean_Dec_Time_pulse[i_ton, i_mu0, i_models]    = np.sum((Prob_list_corr_pulse_temp)*t_list) / np.sum((Prob_list_corr_pulse_temp))                                                           # Only correct choices. Note that Mean_Dec_Time does not includes choices supposedly undecided and made at the last moment.


    ## For each model and each t_onset_pulse, fit the psychometric function
    psychometric_params_list_pulse = np.zeros((3 , len(t_onset_list_pulse), len(models_list)))                          # Note that params_pm in Psychometric_fit_P has only 2 fit parameters...
    param_fit_0_pulse = [2. ,1., 0.]                                                                                    # Initial guess for param_pm for Psychometric_fit_P.
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        for i_ton in range(len(t_onset_list_pulse)):
            t_onset_temp = t_onset_list_pulse[i_ton]
            res_temp = minimize(Psychometric_fit_P, param_fit_0_pulse, args = ([Prob_final_corr_pulse[i_ton,:,i_models]]))     #Note that mu_0_list is intrinsically defined in the Psychometric_fit_P function
            psychometric_params_list_pulse[:,i_ton,i_models] = res_temp.x




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
        axP4.plot(mu_0_list, Prob_final_corr[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
        axP4.plot(mu_0_list, Prob_final_corr_pulse[-1,:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use], linestyle="--" )
        axP4.plot(mu_0_list, 0.5 + np.sign(mu_0_list)*0.5*(1. - np.exp(-(np.sign(mu_0_list)*(mu_0_list+psychometric_params_list_pulse[2,-1,i_models])/psychometric_params_list_pulse[0,-1,i_models]) **psychometric_params_list_pulse[1,-1,i_models])) , color=color_list[index_model_2use], label=labels_list[index_model_2use]+'_long' , linestyle=":" )
    axP2.set_xlabel('mu_0 (~coherence)')
    axP4.set_ylabel('Correct Probability')
    axP4.set_title('Correct Probability')
    axP4.legend(loc=4)



    figP.savefig('Pulse_paradigm.pdf')
    np.save( "pulse_paradigm_x.npy", t_onset_list_pulse)
    np.save( "pulse_paradigm_y.npy", psychometric_params_list_pulse[2,:,:]*100./mu_0)







########################################################################################################################
# Duration Paradigm...
if Flag_Duration:
    t_dur_list_duration = np.arange(0.1, T_dur+0.01, 0.1)

    models_list = [0,1,2]                                                                                               # List of models to use. See Setting_list
    coh_skip_threshold = 60.                                                                                            # Threshold value above which data is skipped.
    n_skip_fit_list = np.zeros(len(models_list))                                                                        # Define the number of skipped data based on coh_skip_threshold later on.


    Prob_final_corr_duration  = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(models_list)))
    Prob_final_err_duration   = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(models_list)))
    Prob_final_undec_duration = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(models_list)))
    Mean_Dec_Time_duration    = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(models_list)))




    ## For each models, find the probability to be correct/erred/undec for various mu and t_onset_pulse
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        for i_mu0 in range(len(mu_0_list)):
            mu_2use = mu_0_list[i_mu0]
            for i_tdur in range(len(t_dur_list_duration)):
                t_dur_temp = t_dur_list_duration[i_tdur]
                t_list_temp = t_list                                                                                    # Cutoff time is constant/ indep of T_dur
                (Prob_list_corr_duration_temp, Prob_list_err_duration_temp) = DDM_pdf_general([mu_2use, param_mu_x_list[index_model_2use], param_mu_t_list[index_model_2use], sigma_0, param_sigma_x_list[index_model_2use], param_sigma_t_list[index_model_2use], B, param_B_t_list[index_model_2use], t_dur_temp], index_model_2use, 2)                               # PyDDM
                Prob_list_sum_corr_duration_temp  = np.sum(Prob_list_corr_duration_temp)
                Prob_list_sum_err_duration_temp   = np.sum(Prob_list_err_duration_temp)
                Prob_list_sum_undec_duration_temp = 1. - Prob_list_sum_corr_duration_temp - Prob_list_sum_err_duration_temp

                #Output
                Prob_final_undec_duration[i_tdur, i_mu0, i_models] = Prob_list_sum_undec_duration_temp
                Prob_final_corr_duration[i_tdur, i_mu0, i_models]  = Prob_list_sum_corr_duration_temp + Prob_final_undec_duration[i_tdur, i_mu0, i_models]/2.
                Prob_final_err_duration[i_tdur, i_mu0, i_models]   = Prob_list_sum_err_duration_temp  + Prob_final_undec_duration[i_tdur, i_mu0, i_models]/2.
                Mean_Dec_Time_duration[i_tdur, i_mu0, i_models]    = np.sum((Prob_list_corr_duration_temp+Prob_list_err_duration_temp) *t_list_temp) / np.sum((Prob_list_corr_duration_temp+Prob_list_err_duration_temp))

    ## For each model and each t_dur_duration, fit the psychometric function
    psychometric_params_list_duration = np.zeros((2 , len(t_dur_list_duration), len(models_list)))
    param_fit_0_duration = [2.,0.5]                                                                                     # Initial guess for param_pm for Psychometric_fit_D.
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        for i_tdur in range(len(t_dur_list_duration)):
            t_dur_temp = t_dur_list_duration[i_tdur]
            res_temp = minimize(Psychometric_fit_D, param_fit_0_duration, args = ([Prob_final_corr_duration[i_tdur,:,i_models]]))     #Note that mu_0_list is intrinsically defined in the Psychometric_fit function
            psychometric_params_list_duration[:,i_tdur,i_models] = res_temp.x

        n_skip_fit_list[i_models] = int(np.sum( psychometric_params_list_duration[0,:,i_models]*100./mu_0  > coh_skip_threshold))                                         # First define which how many terms in the pscyhomeric_params_list to skip/ not include in fit. All data that has threshold >100% is removed.


    ## Fit Psychometric Threshold with a decaying exponential + Constant
    param_fit_threshold_duration = [15., 0.2, 0., 100.]                                                                 # Initial guess for param_pm for Psychometric_fit_D.
    threshold_fit_params_list_duration = np.zeros((len(param_fit_threshold_duration), len(models_list)))
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        res_temp_threshold = minimize(Threshold_D_fit, param_fit_threshold_duration, args = (psychometric_params_list_duration[0,:,i_models]*100./mu_0, int(n_skip_fit_list[i_models])))     #Note that mu_0_list is intrinsically defined in the Psychometric_fit function
        threshold_fit_params_list_duration[:,i_models] = res_temp_threshold.x


    figD = plt.figure(figsize=(8,10.5))
    axD1 = figD.add_subplot(311)
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axD1.plot(t_dur_list_duration, psychometric_params_list_duration[0,:,i_models]*100./mu_0, color=color_list[index_model_2use], label=labels_list[index_model_2use] )         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
        axD1.plot(t_dur_list_duration, threshold_fit_params_list_duration[0, i_models] + (threshold_fit_params_list_duration[3, i_models])*(np.exp(-((t_dur_list_duration-threshold_fit_params_list_duration[2, i_models])/threshold_fit_params_list_duration[1, i_models]))), color=color_list[index_model_2use], label=labels_list[index_model_2use], linestyle="--" )         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    axD1.set_xlabel('Stimulation Duration (s)')
    axD1.set_ylabel('Threshold')
    axD1.set_title('Psychometric function Decision Threshold')
    axD1.legend(loc=1)

    axD2 = figD.add_subplot(312)
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axD2.plot(t_dur_list_duration, psychometric_params_list_duration[1,:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    axD2.set_ylabel('Order')
    axD2.legend(loc=1)

    axD3 = figD.add_subplot(313)
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axD3.plot(coh_list, Prob_final_corr_duration[0,  :, i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use]+'_short' )
        axD3.plot(coh_list, Prob_final_corr_duration[-1, :, i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use]+'_long' , linestyle=":" )
        axD3.plot(coh_list, 0.5 + 0.5*(1. - np.exp(-(mu_0_list/psychometric_params_list_duration[0,0,i_models]) **psychometric_params_list_duration[1,0,i_models])) , color=color_list[index_model_2use], linestyle="--")#, label=labels_list[index_model_2use]+'_short_F')
        axD3.plot(coh_list, 0.5 + 0.5*(1. - np.exp(-(mu_0_list/psychometric_params_list_duration[0,-1,i_models])**psychometric_params_list_duration[1,-1,i_models])), color=color_list[index_model_2use], linestyle="-.")#, label=labels_list[index_model_2use]+'_long_F' )
    axD3.set_xlabel('Coherence')
    axD3.set_ylabel('Probability')
    axD3.set_title('Correct Probability')
    axD3.legend(loc=4)


    figD.savefig('Duration_paradigm.pdf')
    np.save( "Duration_paradigm_x.npy", t_dur_list_duration)
    np.save( "Duration_paradigm_y.npy", psychometric_params_list_duration[0,:,:]*100./mu_0)


############################################################    # Param scan with a range of OU parameters


    OU_pos_range = [0.1*param_mu_x_OUpos,0.6*param_mu_x_OUpos, 1.*param_mu_x_OUpos, 1.5*param_mu_x_OUpos, 2.0*param_mu_x_OUpos]
    OU_neg_range = [0.5*param_mu_x_OUneg,0.8*param_mu_x_OUneg, 1.*param_mu_x_OUneg, 1.2*param_mu_x_OUneg]


    Prob_final_corr_duration_scan_OUpos  = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(OU_pos_range)))
    Prob_final_err_duration_scan_OUpos   = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(OU_pos_range)))
    Prob_final_undec_duration_scan_OUpos = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(OU_pos_range)))
    Mean_Dec_Time_duration_scan_OUpos    = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(OU_pos_range)))
    Prob_final_corr_duration_scan_OUneg  = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(OU_neg_range)))
    Prob_final_err_duration_scan_OUneg   = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(OU_neg_range)))
    Prob_final_undec_duration_scan_OUneg = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(OU_neg_range)))
    Mean_Dec_Time_duration_scan_OUneg    = np.zeros((len(t_dur_list_duration), len(mu_0_list), len(OU_neg_range)))

    index_model_OUpos = 1
    index_model_OUneg = 2

    ## For each model, find the probability to be correct/erred/undec for various mu and t_onset_pulse
    for i_OUpos in range(len(OU_pos_range)):
        OU_param_2use = OU_pos_range[i_OUpos]
        for i_mu0 in range(len(mu_0_list)):
            mu_2use = mu_0_list[i_mu0]
            for i_tdur in range(len(t_dur_list_duration)):
                t_dur_temp = t_dur_list_duration[i_tdur]
                t_list_temp = t_list                                                                                    # Cutoff time is constant/ indep of T_dur
                (Prob_list_corr_duration_temp, Prob_list_err_duration_temp) = DDM_pdf_general([mu_2use, OU_param_2use, param_mu_t_list[index_model_OUpos], sigma_0, param_sigma_x_list[index_model_OUpos], param_sigma_t_list[index_model_OUpos], B, param_B_t_list[index_model_OUpos], t_dur_temp], index_model_OUpos, 2)                               # PyDDM
                Prob_list_sum_corr_duration_temp = np.sum(Prob_list_corr_duration_temp)
                Prob_list_sum_err_duration_temp  = np.sum(Prob_list_err_duration_temp)
                Prob_list_sum_undec_duration_temp  = 1. - Prob_list_sum_corr_duration_temp - Prob_list_sum_err_duration_temp

                # Output
                Prob_final_undec_duration_scan_OUpos[i_tdur, i_mu0, i_OUpos] = Prob_list_sum_undec_duration_temp
                Prob_final_corr_duration_scan_OUpos[i_tdur, i_mu0, i_OUpos]  = Prob_list_sum_corr_duration_temp + Prob_list_sum_undec_duration_temp/2.
                Prob_final_err_duration_scan_OUpos[i_tdur, i_mu0, i_OUpos]   = Prob_list_sum_err_duration_temp  + Prob_list_sum_undec_duration_temp/2.
                Mean_Dec_Time_duration_scan_OUpos[i_tdur, i_mu0, i_OUpos]    = np.sum((Prob_list_corr_duration_temp+Prob_list_err_duration_temp) *t_list_temp) / np.sum((Prob_list_corr_duration_temp+Prob_list_err_duration_temp))
    for i_OUneg in range(len(OU_neg_range)):
        OU_param_2use = OU_neg_range[i_OUneg]
        for i_mu0 in range(len(mu_0_list)):
            mu_2use = mu_0_list[i_mu0]
            for i_tdur in range(len(t_dur_list_duration)):
                t_dur_temp = t_dur_list_duration[i_tdur]
                t_list_temp = t_list                                                                                    # Cutoff time is constant/ indep of T_dur
                (Prob_list_corr_duration_temp, Prob_list_err_duration_temp) = DDM_pdf_general([mu_2use, OU_param_2use, param_mu_t_list[index_model_OUneg], sigma_0, param_sigma_x_list[index_model_OUneg], param_sigma_t_list[index_model_OUneg], B, param_B_t_list[index_model_OUpos], t_dur_temp], index_model_OUneg, 2)                               # PyDDM
                Prob_list_sum_corr_duration_temp  = np.sum(Prob_list_corr_duration_temp)
                Prob_list_sum_err_duration_temp  = np.sum(Prob_list_err_duration_temp)
                Prob_list_sum_undec_duration_temp  = 1. - Prob_list_sum_corr_duration_temp - Prob_list_sum_err_duration_temp

                #Outputs...
                Prob_final_undec_duration_scan_OUneg[i_tdur, i_mu0, i_OUneg] = Prob_list_sum_undec_duration_temp
                Prob_final_corr_duration_scan_OUneg[i_tdur, i_mu0, i_OUneg]  = Prob_list_sum_corr_duration_temp + Prob_list_sum_undec_duration_temp/2.
                Prob_final_err_duration_scan_OUneg[i_tdur, i_mu0, i_OUneg]   = Prob_list_sum_err_duration_temp  + Prob_list_sum_undec_duration_temp/2.
                Mean_Dec_Time_duration_scan_OUneg[i_tdur, i_mu0, i_OUneg]    = np.sum((Prob_list_corr_duration_temp+Prob_list_err_duration_temp) *t_list_temp) / np.sum((Prob_list_corr_duration_temp+Prob_list_err_duration_temp))



    ## For each model and each t_dur_duration, fit the psychometric function
    param_fit_0_duration = [2.,0.5]                                                                                                     # Initial guess for param_pm for Psychometric_fit_D.
    psychometric_params_list_duration_scan_OUpos = np.zeros((2 , len(t_dur_list_duration), len(OU_pos_range)))
    psychometric_params_list_duration_scan_OUneg = np.zeros((2 , len(t_dur_list_duration), len(OU_neg_range)))
    for i_tdur in range(len(t_dur_list_duration)):
        t_dur_temp = t_dur_list_duration[i_tdur]
        for i_OUpos in range(len(OU_pos_range)):
            res_temp = minimize(Psychometric_fit_D, param_fit_0_duration, args = ([Prob_final_corr_duration_scan_OUpos[i_tdur,:,i_OUpos]]))     #Note that mu_0_list is intrinsically defined in the Psychometric_fit function
            psychometric_params_list_duration_scan_OUpos[:,i_tdur,i_OUpos] = res_temp.x
        for i_OUneg in range(len(OU_neg_range)):
            res_temp = minimize(Psychometric_fit_D, param_fit_0_duration, args = ([Prob_final_corr_duration_scan_OUneg[i_tdur,:,i_OUneg]]))     #Note that mu_0_list is intrinsically defined in the Psychometric_fit function
            psychometric_params_list_duration_scan_OUneg[:,i_tdur,i_OUneg] = res_temp.x


    ## Fit Psychometric Threshold with a decaying exponential + Constant
    param_fit_threshold_duration = [15., 0.2, 0., 100.]                                                                                                     # Initial guess for param_pm for Psychometric_fit_D.
    n_skip_fit_list_OUpos = np.zeros(len(OU_pos_range))
    n_skip_fit_list_OUneg = np.zeros(len(OU_neg_range))
    threshold_fit_params_list_duration_scan_OUpos = np.zeros((len(param_fit_threshold_duration), len(OU_pos_range)))
    threshold_fit_params_list_duration_scan_OUneg = np.zeros((len(param_fit_threshold_duration), len(OU_neg_range)))
    for i_OUpos in range(len(OU_pos_range)):
        n_skip_fit_OUpos = int(np.sum( psychometric_params_list_duration_scan_OUpos[0,:,i_OUpos]*100./mu_0  > coh_skip_threshold))                                         # First define which how many terms in the pscyhomeric_params_list to skip/ not include in fit. All data that has threshold >100% is removed.
        n_skip_fit_list_OUpos[i_OUpos] = n_skip_fit_OUpos
        res_scan_OUpos = minimize(Threshold_D_fit, param_fit_threshold_duration, args = (psychometric_params_list_duration_scan_OUpos[0,:,i_OUpos]*100./mu_0, int(n_skip_fit_OUpos)))     #Note that mu_0_list is intrinsically defined in the Psychometric_fit function
        threshold_fit_params_list_duration_scan_OUpos[:,i_OUpos] = res_scan_OUpos.x
    for i_OUneg in range(len(OU_neg_range)):
        n_skip_fit_OUneg = int(np.sum( psychometric_params_list_duration_scan_OUneg[0,:,i_OUneg]*100./mu_0  > coh_skip_threshold))                                         # First define which how many terms in the pscyhomeric_params_list to skip/ not include in fit. All data that has threshold >100% is removed.
        n_skip_fit_list_OUneg[i_OUneg] = n_skip_fit_OUneg
        res_scan_OUneg = minimize(Threshold_D_fit, param_fit_threshold_duration, args = (psychometric_params_list_duration_scan_OUneg[0,:,i_OUneg]*100./mu_0, int(n_skip_fit_OUneg)))     #Note that mu_0_list is intrinsically defined in the Psychometric_fit function
        threshold_fit_params_list_duration_scan_OUneg[:,i_OUneg] = res_scan_OUneg.x





    figD_params = plt.figure(figsize=(8,10.5))
    axD_params1 = figD_params.add_subplot(211)
    for i_OUpos in range(len(OU_pos_range)):
        axD_params1.plot(threshold_fit_params_list_duration_scan_OUpos[1, i_OUpos], threshold_fit_params_list_duration_scan_OUpos[0, i_OUpos], color=color_list[3], marker="x")         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    for i_OUneg in range(len(OU_neg_range)):
        axD_params1.plot(threshold_fit_params_list_duration_scan_OUneg[1, i_OUneg], threshold_fit_params_list_duration_scan_OUneg[0, i_OUneg], color=color_list[4], marker="x")         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    ## Also plot the 3 standard cases for illustration
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axD_params1.plot(threshold_fit_params_list_duration[1, i_models], threshold_fit_params_list_duration[0, i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use], marker="o")         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    axD_params1.set_xlabel('Decay Time Constant (s)')
    axD_params1.set_ylabel('Threshold Asymptote (%)')
    axD_params1.set_title('Psychometric function Decision Threshold')
    axD_params1.legend(loc=2)

    axD_params2 = figD_params.add_subplot(212)
    for i_OUpos in range(len(OU_pos_range)):
        axD_params2.plot(t_dur_list_duration, psychometric_params_list_duration_scan_OUpos[0,:,i_OUpos]*100./mu_0, color=color_list[4] )         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
        axD_params2.plot(t_dur_list_duration, threshold_fit_params_list_duration_scan_OUpos[0, i_OUpos] + threshold_fit_params_list_duration_scan_OUpos[3, i_OUpos]*(np.exp(-((t_dur_list_duration-threshold_fit_params_list_duration_scan_OUpos[2, i_OUpos])/threshold_fit_params_list_duration_scan_OUpos[1, i_OUpos]))), color=color_list[3], linestyle="--" )         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    for i_OUneg in range(len(OU_neg_range)):
        axD_params2.plot(t_dur_list_duration, psychometric_params_list_duration_scan_OUneg[0,:,i_OUneg]*100./mu_0, color=color_list[4] )         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
        axD_params2.plot(t_dur_list_duration, threshold_fit_params_list_duration_scan_OUneg[0, i_OUneg] + threshold_fit_params_list_duration_scan_OUneg[3, i_OUneg]*(np.exp(-((t_dur_list_duration-threshold_fit_params_list_duration_scan_OUneg[2, i_OUneg])/threshold_fit_params_list_duration_scan_OUneg[1, i_OUneg]))), color=color_list[4], linestyle="--" )         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    ## Also plot the 3 standard cases for illustration
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axD_params2.plot(t_dur_list_duration, psychometric_params_list_duration[0,:,i_models]*100./mu_0, color=color_list[index_model_2use], label=labels_list[index_model_2use] )         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
        axD_params2.plot(t_dur_list_duration, threshold_fit_params_list_duration[0, i_models] + (threshold_fit_params_list_duration[3, i_models])*(np.exp(-((t_dur_list_duration-threshold_fit_params_list_duration[2, i_models])/threshold_fit_params_list_duration[1, i_models]))), color=color_list[index_model_2use], label=labels_list[index_model_2use], linestyle="--" )         # *100./mu_0 to convert the threshold from rel to mu_0 to coh level.
    axD_params2.set_xlabel('Stimulation Duration (s)')
    axD_params2.set_ylabel('Threshold')
    axD_params2.set_title('Psychometric function Decision Threshold')
    axD_params2.legend(loc=1)


    figD_params.savefig('Duration_paradigm_params_scan.pdf')
    np.save( "Duration_pscan_OUpos.npy", threshold_fit_params_list_duration_scan_OUpos)
    np.save( "Duration_pscan_OUneg.npy", threshold_fit_params_list_duration_scan_OUneg)
    np.save( "Duration_pscan_ref.npy"  , threshold_fit_params_list_duration)
    np.save( "Duration_pscan_t_dur_list.npy", t_dur_list_duration)
    np.save( "Duration_pscan_psychometric.npy", psychometric_params_list_duration[0,:,:]*100./mu_0)





# ########################################################################################################################
# Psychophysical Kerenel (PK). It may be worth it to run PK on computer-cluster resources, parallelizing the trials into separate runs.

if Flag_PK:
    ## Initialization
    n_rep_PK = 1000                                                                                                     # Number of PK runs
    models_list = [0,1,2]                                                                                               # List of models to use. See Setting_list
    mu_0_PK      = mu_0
    coh_list_PK  = np.array([-25.6, -12.8, -6.4, 6.4, 12.8, 25.6])
    mu_0_list_PK = [mu_0_PK*0.01*coh_temp_PK for coh_temp_PK in coh_list_PK]
    dt_mu_PK = 0.05                                                                                                     #[s] Duration of step for each mu in PK

    ### If want to save details for each trial.
    # mu_t_list_PK = np.zeros(int(T_dur/dt_mu_PK), n_rep_PK, len(models_list))
    # Prob_final_corr_PK  = np.zeros((n_rep_PK, len(mu_0_list_PK), len(models_list)))
    # Prob_final_err_PK   = np.zeros((n_rep_PK, len(mu_0_list_PK), len(models_list)))
    # Prob_final_undec_PK = np.zeros((n_rep_PK, len(mu_0_list_PK), len(models_list)))
    # Mean_Dec_Time_PK    = np.zeros((n_rep_PK, len(mu_0_list_PK), len(models_list)))

    PK_Amp       = np.zeros((int(T_dur/dt_mu_PK), len(mu_0_list_PK), len(models_list)))
    PK_n_trials  = np.zeros((int(T_dur/dt_mu_PK), len(mu_0_list_PK), len(models_list)))                                 # Number of trials run in that slot of the matrix. Used for normalization when doing averaging.


    ## Run the trials
    ## For each models, find the probability to be correct/erred/undec for each PK trial.
    t_list_temp = t_list
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        for i_rep_PK in range(n_rep_PK):
            print(i_rep_PK)
            ind_mu_t_list_PK_temp = np.random.randint(len(mu_0_list_PK), size=int(T_dur/dt_mu_PK))
            mu_t_list_PK_temp = np.zeros((int(T_dur/dt_mu_PK)))
            for i_mu_t_PK in range(len(mu_t_list_PK_temp)):
                mu_t_list_PK_temp[i_mu_t_PK] = mu_0_list_PK[ind_mu_t_list_PK_temp[i_mu_t_PK]]
            (Prob_list_corr_PK_temp, Prob_list_err_PK_temp) = DDM_pdf_general([0., param_mu_x_list[index_model_2use], param_mu_t_list[index_model_2use], sigma_0, param_sigma_x_list[index_model_2use], param_sigma_t_list[index_model_2use], B, param_B_t_list[index_model_2use], mu_t_list_PK_temp], index_model_2use, 1)                               # PyDDM
            Prob_list_sum_corr_PK_temp = np.sum(Prob_list_corr_PK_temp)                                                 # No need cumsum, just sum.
            Prob_list_sum_err_PK_temp  = np.sum(Prob_list_err_PK_temp)
            # Prob_list_sum_undec_PK_temp  = 1. - Prob_list_sum_corr_PK_temp - Prob_list_sum_err_PK_temp                # No need for undecided results, as we only want corr - err, while the undecided trials are spread evenly between the 2.
            PK_Amp_temp = Prob_list_sum_corr_PK_temp - Prob_list_sum_err_PK_temp
            for i_t_PK in range(int(T_dur/dt_mu_PK)):
                PK_Amp[i_t_PK, ind_mu_t_list_PK_temp[i_t_PK], i_models] += Prob_list_sum_corr_PK_temp - Prob_list_sum_err_PK_temp
                PK_n_trials[i_t_PK, ind_mu_t_list_PK_temp[i_t_PK], i_models] += 1

            ### If want to save details for each trial.
            # mu_t_list_PK[:, i_rep_PK, i_models] = mu_t_list_PK_temp                                                     #Record the states
            # Prob_final_undec_PK[i_rep_PK, i_mu0, i_models] = Prob_list_cumsum_undec_PK_temp[-1]
            # Prob_final_corr_PK[ i_rep_PK, i_mu0, i_models] = Prob_list_cumsum_corr_PK_temp[-1] + Prob_final_undec_PK[i_rep_PK, i_mu0, i_models]/2.
            # Prob_final_err_PK[  i_rep_PK, i_mu0, i_models] = Prob_list_cumsum_err_PK_temp[-1]  + Prob_final_undec_PK[i_rep_PK, i_mu0, i_models]/2.
            # Mean_Dec_Time_PK[   i_rep_PK, i_mu0, i_models] = np.sum((Prob_list_cumsum_corr_PK_temp+Prob_list_cumsum_err_PK_temp) *t_list_temp) / np.sum((Prob_list_cumsum_corr_PK_temp+Prob_list_cumsum_err_PK_temp))


    PK_Amp_runNorm = PK_Amp/PK_n_trials                                                                                 # Normalize the terms by the number of runs for each t_bin and mu level, for all models.
    PK_Amp_CohNorm = copy.copy(PK_Amp_runNorm)                                                                          # PK_Amp_CohNorm is PK_Amp but divded by mu, for each of them.
    for i_coh_PK in range(len(coh_list_PK)):
        PK_Amp_CohNorm[:,i_coh_PK,:] /= coh_list_PK[i_coh_PK]
    PK_Amp_1D = np.mean(PK_Amp_CohNorm, axis=1)






    # Plots
    t_list_plot_PK = np.arange(0,T_dur, dt_mu_PK)
    figPK = plt.figure(figsize=(8,10.5))
    axPK1 = figPK.add_subplot(211)
    for i_models in range(len(models_list)):
        index_model_2use = models_list[i_models]
        axPK1.plot(t_list_plot_PK, PK_Amp_1D[:,i_models], color=color_list[index_model_2use], label=labels_list[index_model_2use] )
    axPK1.set_xlabel('mu_0 (~coherence)')
    axPK1.set_ylabel('PK Amp')
    axPK1.set_title('PsychoPhysical Amplitude, 1D')
    axPK1.legend(loc=1)

    plt.subplot(212)
    aspect_ratio = (t_list_plot_PK[-1]-t_list_plot_PK[0])/(mu_0_list_PK[-1]-mu_0_list_PK[0])
    plt.imshow(PK_Amp_CohNorm[:,:,0]           , extent=(mu_0_list_PK[0], mu_0_list_PK[-1], t_list_plot_PK[0], t_list_plot_PK[-1]), interpolation='nearest', cmap=matplotlib_cm.gist_rainbow, aspect=aspect_ratio)
    plt.colorbar()
    plt.xlabel("t_bin (s)")
    plt.ylabel("mu_0 (~coherence)")
    plt.title('PsychoPhysical Amplitude, 2D, Control', fontsize=10)


    figPK.savefig('Psychophysical_Kernel.pdf')
    np.save("PK_Amp_2D.npy", PK_Amp)
    np.save("PK_n_runs_2D.npy"    , PK_n_trials)









