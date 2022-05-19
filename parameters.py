# === Simulation-related parameters ====

defaultclock.dt = 0.02*ms # simulation step length [ms]
dt_default = 0.02*ms # simulation step length [ms]
simtime     = 5000.0*ms     # total simulation time [ms]
simulation_clock=Clock(dt=dt_default)   # clock for the time steps. Temporarily used since defaultclock is not functional.


# === Parameters determining model dynamics =====

# network structure
NE = 1600                   # Number of Excitatory Neurons
NI = 400                    # Number of Inhibitory Neurons
f = 0.15                    # Fraction of E-cells in group
N1 = (int)(f*NE)            # Number of Excitatory Neurons in group 1
N2 = (int)(f*NE)            # Number of Excitatory Neurons in group 2
N0=NE-N1-N2                 # Number of Non-selective Excitatory Neurons
wp = 1.84                   # strengthened connection within group
wm = 1.0-f*(wp-1.0)/(1.0-f) # Weakened connection across groups, to preserve total recurrent strength.

# pyramidal cells
Cm_e = 0.5*nF     # [nF] total capacitance
gl_e = 25.0*nS    # [ns] total leak conductance
El_e = -70.0*mV   # [mV] leak reversal potential
Vt_e = -50.0*mV   # [mV] threshold potential
Vr_e = -55.0*mV   # [mV] reset potential
tr_e = 2.0*ms     # [ms] refractory time

# interneurons
Cm_i = 0.2*nF       # [nF] total capacitance
gl_i = 20.0*nS      # [ns] total leak conductance
El_i = -70.0*mV     # [mV] leak reversal potential
Vt_i = -50.0*mV     # [mV] threshold potential
Vr_i = -55.0*mV     # [mV] reset potential
tr_i = 1.0*ms       # [ms] refractory time

# AMPA receptor
E_ampa = 0.0*mV             # [mV] synaptic reversial potential
t_ampa = 2.0*ms             # [ms] exponential decay time constant
g_ext_e = 2.07*nS           # [nS] conductance from external to pyramidal cells
g_ext_i = 1.62*nS           # [nS] conductance from external to interneurons
g_ampa_i = 0.04*nS          # [nS] conductance from pyramidal to interneurons
g_ampa_e = 0.05*nS          # [nS] conductance from pyramidal to pyramidal cells
wext_e = g_ext_e/g_ampa_e   # normalized external conductance to E cells
wext_i = g_ext_i/g_ampa_i   # normalized external conductance to I cells

# GABA receptor
E_gaba = -70.0*mV   # [mV] synaptic reversial potential
t_gaba = 5.0*ms     # [ms] exponential decay time constant
g_gaba_e = 1.3*nS   # [nS] conductance to pyramidal cells
g_gaba_i = 1.0*nS   # [nS] conductance to interneurons

# NMDA receptor
E_nmda = 0.0*mV     # [mV] synaptic reversial potential
t_nmda = 100.0*ms   # [ms] decay time of NMDA currents
t_x = 2.0*ms        # [ms] controls the rise time of NMDAR channels
alpha = 0.5*kHz     # [kHz] controls the saturation properties of NMDAR channels

# Conductance strength between Pyramidal neurons groups, for control/ reduced EE/ reduced EI
g_nmda_e = 0.165* (1. - 0.*0.02) *nS             # Control or reduced EE
g_nmda_i = 0.13 * (1. - 0.*0.03) *nS             # Control or reduced EI

a=0.062/mV  # control the voltage dependance of NMDAR channel
b=1/3.57  # control the voltage dependance of NMDAR channel ([Mg2+]=1mM )




########################################################################################################################
### External Input.
# For pulse paradigm, change tpulse_start (and make sure pk>0). Otherwise set pk=0.
# For duration paradigm, change ts_stop.
# Pychophysical kernel in alternative code.

fext     = 2400.0*Hz    # Background input
ts_start = 1.0*second   # stimulus onset
ts_stop  = 3.0*second   # stimulus offset
mu       = 38.0*Hz      # stimulus strength
pk       = 1.*0.15      # Pulse strength, relative to/ percentage of mu. Set to 0 if not in pulse paradigm.
#dt_stim       = 100.*ms               # Time step of updating stimulus (should be the highest common denominator between tc_start and tr_stop)

## Here set pulse effectively 0, pulse paragdigm will change the definition elsewhere.
tpulse_start = 0.0*second
tpulse_dur   = 0.1*second
tpulse_stop  = tpulse_start + tpulse_dur


coherence = 0.  # Coherence.
upstream_deficit_coeff = 1. # deficit from upstream sensory regions. <=1. 1= no deficit, 0=full deficit.

stimulus1= 'fext + (0.5*(sign(t-ts_start)+1))*(0.5*(sign(ts_stop-t)+1))*mu*(1.0 + upstream_deficit_coeff*(0.01*coherence + pk*(0.5*(sign(t-tpulse_start)+1))*(0.5*(sign(tpulse_stop-t)+1))))'
stimulus2= 'fext + (0.5*(sign(t-ts_start)+1))*(0.5*(sign(ts_stop-t)+1))*mu*(1.0 - upstream_deficit_coeff*(0.01*coherence + pk*(0.5*(sign(t-tpulse_start)+1))*(0.5*(sign(tpulse_stop-t)+1))))'

dt_psth = 0.001*second

