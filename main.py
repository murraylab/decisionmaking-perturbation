"""
A model of decision making and E/I perturbation from Lam et al. (JNeurosci 2022), implemented in the Brian2 simulator.
Note that parameter scans are done on the high-performance computer cluster at Yale.
Those part of the codes specify parameters after executing parameters.py in main.py, with no additional change to current codes.
Such codes are not included as they are specfic to slurm (fairly common, but not universal to all clusters).
Please feel free to contact the authors for such aditional parts codes.
Contact: Norman Lam (normanlam1217@gmail.com)
"""

import matplotlib
matplotlib.use('Agg') # to generate figures using the cluster
import os
import sys
sys.path.append("/net/murray/nhl8/Modules/brian-1.4.1/")
import numpy
from numpy.random import seed
from pylab import *
import argparse
from brian2 import *
import analysis


#-----------------------------------------------------------------------------------------
### Load default parameters ###
# execfile('parameters.py')                                                                                             # python 2
exec(open("parameters.py").read())                                                                                      # python 3


# Dynamical equations
eqs_e = '''
dv/dt = (-gl_e*(v-El_e)-g_gaba_e*s_gaba*(v-E_gaba)-g_ampa_e*s_ampa*(v-E_ampa)-g_nmda_e*s_tot*(v-E_nmda)/(1+b*exp(-a*v)))/Cm_e: volt (unless refractory)
ds_gaba/dt = -s_gaba/t_gaba : 1
ds_ampa/dt = -s_ampa/t_ampa : 1
ds_nmda/dt = -s_nmda/t_nmda+alpha*x*(1-s_nmda) : 1
dx/dt = -x/t_x : 1
s_tot : 1
'''
eqs_i = '''
dv/dt = (-gl_i*(v-El_i)-g_gaba_i*s_gaba*(v-E_gaba)-g_ampa_i*s_ampa*(v-E_ampa)-g_nmda_i*s_tot*(v-E_nmda)/(1+b*exp(-a*v)))/Cm_i: volt (unless refractory)
ds_gaba/dt = -s_gaba/t_gaba : 1
ds_ampa/dt = -s_ampa/t_ampa : 1
s_tot : 1
'''

Pe = NeuronGroup(NE, eqs_e, threshold='v>Vt_e', reset='v=Vr_e', refractory=tr_e, order=2, clock=simulation_clock)
Pe.v = El_e
Pe.s_ampa = 0
Pe.s_gaba = 0
Pe.s_nmda = 0
Pe.x = 0

Pi = NeuronGroup(NI, eqs_i, threshold='v>Vt_i', reset='v=Vr_i', refractory=tr_i, order=2, clock=simulation_clock)
Pi.v = El_i
Pi.s_ampa = 0
Pi.s_gaba = 0

# create 3 excitatory subgroups: 1 & 2 are selective to motion direction, 0 is not
Pe0 = Pe[:N0]
Pe1 = Pe[N0:(N0+N1)]
Pe2 = Pe[(N0+N1):]

# external Poisson input
PGi = PoissonGroup(NI, fext, clock=simulation_clock)
PG0 = PoissonGroup(N0, fext, clock=simulation_clock)
PG1 = PoissonGroup(N1, stimulus1, clock=simulation_clock)
PG2 = PoissonGroup(N2, stimulus2, clock=simulation_clock)


selfnmda = Synapses(Pe, Pe, 'w:1', on_pre='x_post+=w', clock=simulation_clock) # Update NMDA gating variables of E-cells (pre-synaptic)
selfnmda.connect(j='i')
selfnmda.w = 1.
selfnmda.delay = '0.5*ms'

# Poisson Synapses
Cp1 = Synapses(PG1, Pe1, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update external AMPA gating variables of Group 1 E-cells (pre-synaptic)
Cp1.connect(j='i')
Cp1.w = wext_e
Cp2 = Synapses(PG2, Pe2, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update external AMPA gating variables of Group 2 E-cells (pre-synaptic)
Cp2.connect(j='i')
Cp2.w = wext_e
Cp0 = Synapses(PG0, Pe0, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update external AMPA gating variables of non-selective E-cells (pre-synaptic)
Cp0.connect(j='i')
Cp0.w = wext_e
Cpi = Synapses(PGi, Pi, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update external AMPA gating variables of I-cells (pre-synaptic)
Cpi.connect(j='i')
Cpi.w = wext_i

# Recurrent Synapses (non-NMDA)
Cie = Synapses(Pi, Pe, 'w:1', on_pre='s_gaba_post+=w', clock=simulation_clock) # Update GABA gating variables of all E-cells (pre-synaptic)
Cie.connect()
Cie.w = 1.
Cie.delay = '0.5*ms'
Cii = Synapses(Pi, Pi, 'w:1', on_pre='s_gaba_post+=w', clock=simulation_clock) # Update GABA gating variables of I-cells (pre-synaptic)
Cii.connect()
Cii.w = 1.
Cii.delay = '0.5*ms'
Cei = Synapses(Pe, Pi, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables of I-cells (pre-synaptic)
Cei.connect()
Cei.w = 1.
Cei.delay = '0.5*ms'
C00 = Synapses(Pe0, Pe0, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables within non-selective E-cells (pre-synaptic)
C00.connect()
C00.w = 1.
C00.delay = '0.5*ms'
C10 = Synapses(Pe1, Pe0, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from group 1 to non-selective E-cells (pre-synaptic)
C10.connect()
C10.w = 1.
C10.delay = '0.5*ms'
C20 = Synapses(Pe2, Pe0, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from group 2 to non-selective E-cells (pre-synaptic)
C20.connect()
C20.w = 1.
C20.delay = '0.5*ms'
C01 = Synapses(Pe0, Pe1, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from non-selective to group 1 E-cells (pre-synaptic)
C01.connect()
C01.w = wm
C01.delay = '0.5*ms'
C11 = Synapses(Pe1, Pe1, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables within group 1 E-cells (pre-synaptic)
C11.connect()
C11.w = wp
C11.delay = '0.5*ms'
C21 = Synapses(Pe2, Pe1, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from group 2 to non-selective E-cells (pre-synaptic)
C21.connect()
C21.w = wm
C21.delay = '0.5*ms'
C02 = Synapses(Pe0, Pe2, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from non-selective to group 2 E-cells (pre-synaptic)
C02.connect()
C02.w = wm
C02.delay = '0.5*ms'
C12 = Synapses(Pe1, Pe2, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables from group 1 to group 2 E-cells (pre-synaptic)
C12.connect()
C12.w = wm
C12.delay = '0.5*ms'
C22 = Synapses(Pe2, Pe2, 'w:1', on_pre='s_ampa_post+=w', clock=simulation_clock) # Update AMPA gating variables within group 2 E-cells (pre-synaptic)
C22.connect()
C22.w = wp
C22.delay = '0.5*ms'


# Calculate NMDA contributions (Post-synaptic)
@network_operation(clock=simulation_clock, when='start')
def update_nmda():
    s_NMDA1 = np.sum(Pe1.s_nmda)
    s_NMDA2 = np.sum(Pe2.s_nmda)
    s_NMDA0 = np.sum(Pe0.s_nmda)
    Pe1.s_tot = (wp*s_NMDA1+wm*s_NMDA2+wm*s_NMDA0)
    Pe2.s_tot = (wm*s_NMDA1+wp*s_NMDA2+wm*s_NMDA0)
    Pe0.s_tot = (s_NMDA1+s_NMDA2+s_NMDA0)
    Pi.s_tot  = (s_NMDA1+s_NMDA2+s_NMDA0)

# initiating monitors
monit_P1 = SpikeMonitor(Pe1, record=True)
monit_P2 = SpikeMonitor(Pe2, record=True)

# Run Simulations
net = Network(Pe,Pe0,Pe1,Pe2,Pi,PG0,PG1,PG2,PGi,selfnmda,Cp0,Cp1,Cp2,Cpi,Cie,Cii,Cei,C00,C10,C20,C01,C11,C21,C02,C12,C22,update_nmda,monit_P1,monit_P2)
net.run(simtime, report='text')

# Save data
np.savetxt('spikes_s1.dat',np.column_stack((monit_P1.i, monit_P1.t)))
np.savetxt('spikes_s2.dat',np.column_stack((monit_P2.i, monit_P2.t)))




########################################################################################################################
### Result of individual simulation.


#Define list of time.
dt_psth = 0.001*second
tpts_0 = int(np.round((simtime-0.*second)/dt_psth))+1
t_vec_list = np.linspace(0., simtime/second, tpts_0)






### Summarize simulated data
filename_spikes_Pe1  = ("spikes_s1.dat")   # Set file name for soma spikes.
filename_spikes_Pe2  = ("spikes_s2.dat")   # Set file name for soma spikes.

spikes_Pe1_temp  = analysis.load_raster_spikes(filename_spikes_Pe1)
r_Pe1_temp       = analysis.psth(spikes_Pe1_temp[:,1], t_vec=t_vec_list, t_start=0., t_end=simtime/second, filter_fn=analysis.filter_exp)/N1

spikes_Pe2_temp  = analysis.load_raster_spikes(filename_spikes_Pe2)
r_Pe2_temp       = analysis.psth(spikes_Pe2_temp[:,1], t_vec=t_vec_list, t_start=0., t_end=simtime/second, filter_fn=analysis.filter_exp)/N2




win_12=0
RT_win=0.
# Define winner as the first of the 2 group who reaches 15Hz first.
for ii in range(len(r_Pe1_temp)):
    if r_Pe1_temp[ii] >= 15.0 :
       win_12 = 1
       RT_win = t_vec_list[ii]
       break
    elif r_Pe2_temp[ii] >= 15.0 :
       win_12 = 2
       RT_win = t_vec_list[ii]
       break
    else :
        win_12 = 0




## Save various plots and data.
#Save Scatter Plots for P cells.
# if not os.path.exists(savepath+'spikes_scatter_E12.pdf'):      						  #Create only if path does not exist
if True:
    fig1 = plt.figure(figsize=(8,10.5))
    n_skip=8 # sparsen scatter plots
    ### Spike rastergram for E1 ###
    ax11 = fig1.add_subplot(311)
    ax11.scatter(spikes_Pe1_temp[:,1][::n_skip],spikes_Pe1_temp[:,0][::n_skip],\
                marker='.',c='red',edgecolor='none',alpha=0.5)
    ax11.set_xlim(0,simtime/second)
    ax11.set_ylabel('Neuron label, E1 (deg)')
    ax11.set_xlabel('Time (s)')

    ### Spike rastergram for E2 ###
    ax12 = fig1.add_subplot(312)
    ax12.scatter(spikes_Pe2_temp[:,1][::n_skip],spikes_Pe2_temp[:,0][::n_skip],\
                marker='.',c='blue',edgecolor='none',alpha=0.5)
    ax12.set_xlim(0,simtime/second)
    ax12.set_ylabel('Neuron label, E2 (deg)')
    ax12.set_xlabel('Time (s)')

    ### Firing rate profiles for P cells###
    ax13 = fig1.add_subplot(313)
    ax13.plot(t_vec_list,r_Pe1_temp,c='red' ,lw=2,label='E1')
    ax13.plot(t_vec_list,r_Pe2_temp,c='blue',lw=2,label='E2')
    ax13.set_xlabel('Time (s)')
    ax13.set_ylabel('Population Firing rate (Hz)')
    ax13.set_xlim(0,simtime/second)
    ax13.legend(frameon=False    )
    fig1.savefig('spikes_scatter_E12.pdf')


np.savetxt("r_smooth_E1.txt", r_Pe1_temp)
np.savetxt("r_smooth_E2.txt", r_Pe2_temp)
np.savetxt("win12_RTwin.txt", (win_12, RT_win))