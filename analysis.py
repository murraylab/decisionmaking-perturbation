import numpy as np


def load_raster(filename):
    raster = np.genfromtxt(filename,delimiter='',invalid_raise=False)
    return raster

def load_raster_spikes(filename):
    raster = np.genfromtxt(filename,delimiter='',invalid_raise=False)
    return raster



#-----------------------------------------------------------------------------------------------------------------------
### Filter and PSTH
#def filter_gauss(t,tau=0.1):
def filter_gauss(t,tau=0.0265):
    z = 1./(np.sqrt(2*np.pi)*tau)*np.exp(-t**2./(2.*tau**2.))
    return z

def filter_exp(t,tau=0.02):
    z = np.zeros(np.shape(t))
    ind1 = np.nonzero(t<0.)
    z[ind1] = 0.
    ind2 = np.nonzero(t>=0.)
    z[ind2] = 1./tau*np.exp(-t[ind2]/tau)
    return z

def filter_box(t,tau=0.1,t_center=0):
    z = 1./tau*(np.abs(t-t_center) <= tau/2.)
    return z

def filter_alpha(t,tau=0.1,t_center=0):
    z = 0. + (t-t_center)/tau*np.exp(-t/tau)*(t>=0.)
    return z




def psth(spikes_times, t_vec=None, t_start = 0, t_end = None, dt_psth = 0.001, filter_fn = filter_exp, params_filter={},duty_cycle=1., seed=None):

    if t_end == None:
        t_end = np.ceil(np.max([np.max(np.hstack((spikes_i,[t_start]))) for spikes_i in spikes]))

    if t_vec is None:
        tpts = int(np.round((t_end-t_start)/dt_psth))+1
        t_vec = np.linspace(t_start, t_end, tpts)
    else:
        tpts = len(t_vec)

    filt_t_vec = np.arange(-(t_end-t_start),(t_end-t_start),dt_psth)

    filt_vec = filter_fn(filt_t_vec,**params_filter)

    filt_vec /= np.sum(filt_vec)

    fr = np.zeros(len(t_vec))

    if seed is None:
        seed=np.random.randint(duty_cycle)

    relevant_indices = np.nonzero((spikes_times >= t_start) & (spikes_times <= t_end) &
                                    (np.mod(spikes_times+seed,duty_cycle)<=1.0001))[0]

    relevant_spikes_times = spikes_times[relevant_indices]
    for k_spike in range(len(relevant_spikes_times)):
        t_spike = relevant_spikes_times[k_spike]
        spike_pt = np.argmin(np.abs(t_spike - t_vec))
        fr += np.roll(filt_vec,-tpts+spike_pt)[:tpts]                                                                   # Adding part of the filter corresponding to the impact after the spike
        fr += np.roll(filt_vec,-tpts+spike_pt)[-tpts:][::-1]                                                            # Adding part of the filter corresponding to the impact before the spike. 0 for casual filters (exponential, alpha etc)
    return fr/dt_psth





