import os
import sys
sys.path.append('/pool/asha0/SCIENCE/csalt/')
import numpy as np
import scipy.constants as sc
from csalt.helpers import *
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt

# filenames
raw_MS = 'ALMA-BLC_244kHz_SAMPLED'			# pre-regridding
reg_MS = raw_MS + '.REGRID0_linear'			# post-regridding
truth_MS = 'TRUTH'					# oversampled truth
truth_resampled_MS = 'TRUTH.ALMA-BLC_244kHz.REGRID0'	# reampled truth

# choose a random EB and visibility index to show
EB, ix = 2, 10000


# ------------------

# load files
raw_dict = read_MS('storage/'+raw_MS+'.ms')
reg_dict = read_MS('storage/'+reg_MS+'.ms')
tru_dict = read_MS('storage/'+truth_MS+'.ms')
res_dict = read_MS('storage/'+truth_resampled_MS+'.ms')

# Nvis per EB
Nvis_per_EB = [raw_dict[str(j)].nvis for j in range(raw_dict['Nobs'])]

# which timestamp ID
_stamp = int(raw_dict[str(EB)].tstamp[ix])

# LSRK velocities
raw_vel = sc.c * (1. - raw_dict[str(EB)].nu_LSRK[_stamp,:] / 230.538e9)
reg_vel = sc.c * (1. - reg_dict['0'].nu_LSRK[0,:] / 230.538e9)
tru_vel = sc.c * (1. - tru_dict[str(EB)].nu_LSRK[_stamp,:] / 230.538e9)
res_vel = sc.c * (1. - res_dict['0'].nu_LSRK[0,:] / 230.538e9)

# vis spectra
raw_vis = raw_dict[str(EB)].vis[0,:,ix]
reg_vis = reg_dict['0'].vis[0,:,int(np.sum(Nvis_per_EB[:EB])+ix)]
tru_vis = tru_dict[str(EB)].vis[0,:,ix]
res_vis = res_dict['0'].vis[0,:,int(np.sum(Nvis_per_EB[:EB])+ix)]

fig, ax = plt.subplots()

ax.plot(raw_vel, raw_vis, '--C0+', lw=2, alpha=0.5, label='observed')
ax.plot(reg_vel, reg_vis, '-C0o', lw=2, label='regridded')
ax.plot(tru_vel, tru_vis, '--C1', lw=2, alpha=0.5, label='truth')
ax.plot(res_vel, res_vis, '-C1o', lw=2, label='resampled truth')

ax.legend()

plt.show()

