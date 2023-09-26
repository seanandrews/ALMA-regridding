import os
import sys
sys.path.append('/pool/asha0/SCIENCE/csalt/')
from csalt.model import *
from csalt.helpers import *
import numpy as np
import scipy.constants as sc
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import casatools
from casatasks import (split, mstransform, concat)
from regrid_interpolators import *
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt

# template initial (not regridded) input file
init_MS = 'storage/ALMA-BLC_244kHz_SAMPLED'

# template regridded input file
in_MS = 'storage/ALMA-BLC_244kHz_SAMPLED.REGRID0_linear'

# output filename
out_MS = 'storage/TRUTH.ALMA-BLC_244kHz.REGRID0'

# truth filename
tru_MS = 'storage/TRUTH'

# spectral response function to use
SRF_ = 'ALMA'


#######################

def BLC_SRF(dnu_native, dnu_sampled=2e3, N_over=25):

    # the over-sampling factor for input spectra
    f_oversample = dnu_native / dnu_sampled

    # number of over-sampled channels needed to sample N_over native channels
    nchan = int(np.round(N_over * f_oversample))

    # channel frequency grid
    chix = np.arange(nchan)
    xch = (chix - np.mean(chix)) / f_oversample

    # compute the SRF on the channel frequency grid
    srf = 0.50 * np.sinc(xch) + \
          0.25 * (np.sinc(xch - 1) + np.sinc(xch + 1))

    return srf / np.sum(srf)


# Load the original (native) MS file into a dictionary
orig_dict = read_MS(init_MS+'.ms')

# Load the truth MS file into a dictionary
t_dict = read_MS(tru_MS+'.ms')

# copy the regridded MS into output file
os.system('cp -r '+in_MS+'.ms '+out_MS+'.ms')
o_dict = read_MS(out_MS+'.ms')

# Until someone shows me how to do this differently, we'll just do the painful
# thing and split / transform in a loop over the EBs, then re-concatenate
o_MSlist = []
stamp_ctr = 0
for EB in range(t_dict['Nobs']):

    # compute the SRF kernel
    dnu_native = np.diff(orig_dict[str(EB)].nu_TOPO)[0]
    dnu_truth = np.diff(t_dict[str(EB)].nu_TOPO)[0]
    kernel = BLC_SRF(dnu_native, dnu_sampled=dnu_truth)

    # cycle over the timestamps and process the *true* visibility spectra
    for j in range(t_dict[str(EB)].nstamps):

        # indices corresponding to this timestamp
        oixl = np.min(np.where(o_dict['0'].tstamp == j+stamp_ctr))
        oixh = np.max(np.where(o_dict['0'].tstamp == j+stamp_ctr)) + 1
        tixl = np.min(np.where(t_dict[str(EB)].tstamp == j))
        tixh = np.max(np.where(t_dict[str(EB)].tstamp == j)) + 1
        print(j, stamp_ctr, oixl, oixh, tixl, tixh)

        # convert true visibilities into a useable array format
        tvis = np.empty((t_dict[str(EB)].npol, t_dict[str(EB)].nchan, 
                         tixh-tixl, 2))
        tvis[0,:,:,0] = t_dict[str(EB)].vis[0,:,tixl:tixh].real
        tvis[1,:,:,0] = t_dict[str(EB)].vis[1,:,tixl:tixh].real
        tvis[0,:,:,1] = t_dict[str(EB)].vis[0,:,tixl:tixh].imag
        tvis[1,:,:,1] = t_dict[str(EB)].vis[1,:,tixl:tixh].imag

        # convolve the true spectra with the SRF for this timestamp
        tvis_conv = convolve1d(tvis, kernel, axis=1, mode='nearest')

        # revert back to complex format
        _tvis = tvis_conv[:,:,:,0] + 1j * tvis_conv[:,:,:,1]

        # interpolate the oversampled true spectra onto the output channels
        fint = interp1d(t_dict[str(EB)].nu_LSRK[j,:], _tvis, axis=1,
                        kind='cubic', fill_value='extrapolate')
        o_dict['0'].vis[:,:,oixl:oixh] = fint(o_dict['0'].nu_LSRK[j,:])

    stamp_ctr += t_dict[str(EB)].nstamps
        
# write out the MS file
write_MS(o_dict, outfile=out_MS+'.ms', direct_file=True)
