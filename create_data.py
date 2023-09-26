import os
import sys
sys.path.append('/pool/asha0/SCIENCE/csalt/')
import numpy as np
from csalt.model import *
from csalt.helpers import *
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'


# setup
SRF_ = 'ALMA'
dnu_ = 244e3
name = 'ALMA-BLC_244kHz'


#------------------------------------------------------------------------------

# Instantiate a csalt model
cm = model('CSALT0', path='/pool/asha0/SCIENCE/csalt/')

# Create an empty MS from scratch
sdir = 'storage/'
cdir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'
cm.template_MS(sdir+'templates/template_'+name+'.ms',
               config=[cdir+'alma.cycle8.4.cfg',
                       cdir+'alma.cycle8.7.cfg',
                       cdir+'alma.cycle8.7.cfg'],
               t_total=['30min', '60min', '60min'], t_integ='30s', 
               observatory='ALMA', 
               date=['2025/03/01', '2025/05/20', '2025/05/20'], 
               HA_0=['-0.25h', '-1.0h', '0.0h'],
               restfreq=230.538e9, dnu_native=dnu_, V_tune=0.7e3, V_span=2.5e3,
               RA='16:00:00.00', DEC='-30:00:00.00')

# Get the data dictionary from the empty MS
ddict = read_MS(sdir+'templates/template_'+name+'.ms')

# Set the CSALT model parameters
pars = np.array([
                   45,  # incl (deg)
                   60,  # PA (deg), E of N to redshifted major axis
                  1.0,  # Mstar (Msun)
                  300,  # R_out (au)
                  0.3,  # emission height z_0 (") at r = 1"
                  1.0,  # phi for z(r) = z_0 * (r / 1)**phi
                  150,  # Tb_0 at r = 10 au (K)
                 -0.5,  # q for Tb(r) = Tb_0 * (r / 10 au)**q
                   20,  # maximum Tb for back surface of disk (~ Tfreezeout)
                  297,  # linewidth at r = 10 au (m/s)
                  3.0,  # log(tau_0) at 10 au
                   -1,  # p for tau(r) = tau_0 * (r / 10 au)**p
                  0e3,  # systemic velocity (m/s)
                    0,  # RA offset (")
                    0   # DEC offset (")
                     ])

# Generate the SAMPLED and NOISY visibility spectra
if SRF_ == 'ALMA-WSU':
    x_scale = 1.2
else:
    x_scale = 1.0
noise = 7.3 * np.sqrt(30.5e3 / dnu_) / x_scale
fixed_kw = {'FOV': 5.11, 'Npix': 512, 'dist': 150, 'Nup': 3, 
            'doppcorr': 'exact', 'SRF': SRF_, 'noise_inject': noise}
sampl_mdict, noisy_mdict = cm.modeldict(ddict, pars, kwargs=fixed_kw)
write_MS(sampl_mdict, outfile=sdir+name+'_SAMPLED.ms')
write_MS(noisy_mdict, outfile=sdir+name+'_NOISY.ms')
