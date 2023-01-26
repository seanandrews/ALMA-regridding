"""
This should work with an external CASA call, like

casa --nologger --nologfile -c regrid_script.py

once you've appropriately edited the top matter.
"""

import os, sys
import numpy as np
execfile('test_interpolators.py')


# input data file
in_MS = 'data/almadev_blc1_pure.DATA.ms'

# output data file
out_MS = 'data/test.ms'

# regrid parameters
chanstart = '-4.2km/s'
chanwidth = '0.2km/s'
nchan = 40
restfreq = '230.538GHz'
imethod = 'linear'	


#######################

### Design Output MS and Get Relevant Quantities
# Make an MS with the appropriate output structure
os.system('rm -rf '+out_MS)
mstransform(vis=in_MS, outputvis=out_MS, datacolumn='data',
            regridms=True, mode='velocity', nchan=nchan, start=chanstart,
            width=chanwidth, restfreq=restfreq, outframe='LSRK')

# Load the output dataset
tb.open(out_MS)
data_ = np.squeeze(tb.getcol('DATA'))
tb.close()

# Get the output LSRK frequencies, nu_
tb.open(out_MS+'/SPECTRAL_WINDOW')
nu_ = np.squeeze(tb.getcol('CHAN_FREQ'))
tb.close()




### Extract Relevant Quantities from Input MS
# Load the input dataset
tb.open(in_MS)
_data = np.squeeze(tb.getcol('DATA'))
_times = tb.getcol('TIME')
tb.close()

# Index the timestamps
tstamps = np.unique(_times)
tstamp_ID = np.empty_like(_times)
for istamp in range(len(tstamps)):
    tstamp_ID[_times == tstamps[istamp]] = istamp
nstamps = len(tstamps)

# Acquire the input TOPO frequencies
tb.open(in_MS+'/SPECTRAL_WINDOW')
nu_TOPO = np.squeeze(tb.getcol('CHAN_FREQ'))
tb.close()

# Compute the input LSRK frequencies for each timestamp, _nu
_nu = np.empty((len(nu_TOPO), len(tstamps)))
ms.open(in_MS)
for istamp in range(len(tstamps)):
    _nu[:,istamp] = ms.cvelfreqs(mode='channel', outframe='LSRK',
                                 obstime=str(tstamps[istamp])+'s')
ms.close()




### Interpolate (in a loop for each timestamp)
for itime in range(len(tstamps)):
    ixl = np.min(np.where(tstamp_ID == itime))
    ixh = np.max(np.where(tstamp_ID == itime)) + 1
    cmd = 'interp_'+imethod+'(_nu[:,itime], _data[:,:,ixl:ixh], nu_, axis=1)'
    data_[:,:,ixl:ixh] = eval(cmd)




### Return the Interpolated Data into the Output MS
tb.open(out_MS, nomodify=False)
tb.putcol('DATA', data_)
tb.flush()
tb.close()
