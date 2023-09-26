import os
import sys
sys.path.append('/pool/asha0/SCIENCE/csalt/')
from csalt.model import *
from csalt.helpers import *
import numpy as np
import scipy.constants as sc
import casatools
from casatasks import (split, mstransform, concat)
from regrid_interpolators import *
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'
import matplotlib.pyplot as plt


# input data file
in_MS = 'storage/ALMA-BLC_244kHz_SAMPLED'

# interpolation method
imeth = 'linear'

# output channel grid
out_grid = '0'			# just a label for this specific channel grid
chanstart = '-1.37km/s'		#'-3.47km/s'
chanwidth = '0.32km/s'
nchan = 13
restfreq = '230.538GHz'


# output data file
out_MS = in_MS+'.REGRID'+out_grid+'_'+imeth




#######################

# Load the input MS file into a dictionary
i_dict = read_MS(in_MS+'.ms')

# Until someone shows me how to do this differently, we'll just do the painful
# thing and split / transform in a loop over the EBs, then re-concatenate
o_MSlist = []
for EB in range(i_dict['Nobs']):

    # split out this EB only
    os.system('rm -rf '+in_MS+'_EB'+str(EB)+'.ms*')
    split(vis=in_MS+'.ms', outputvis=in_MS+'_EB'+str(EB)+'.ms', 
          datacolumn='data', spw=str(EB))

    # mstransform into desired grid
    os.system('rm -rf '+out_MS+'_EB'+str(EB)+'.ms*')
    mstransform(vis=in_MS+'_EB'+str(EB)+'.ms', 
                outputvis=out_MS+'_EB'+str(EB)+'.ms',
                datacolumn='data', regridms=True, mode='velocity',
                nchan=nchan, start=chanstart, width=chanwidth, 
                restfreq=restfreq, outframe='LSRK')

    # read in the regridded MS for this EB into a dictionary
    o_dict = read_MS(out_MS+'_EB'+str(EB)+'.ms')

    # cycle over the timestamps to interpolate onto the regridded LSRK channels
    for j in range(i_dict[str(EB)].nstamps):
        ixl = np.min(np.where(i_dict[str(EB)].tstamp == j))
        ixh = np.max(np.where(i_dict[str(EB)].tstamp == j)) + 1
        cmd = "interp_"+imeth+"(i_dict[str(EB)].nu_LSRK[j,:], "+\
              "i_dict[str(EB)].vis[:,:,ixl:ixh], o_dict['0'].nu_TOPO, axis=1)"
        o_dict['0'].vis[:,:,ixl:ixh] = eval(cmd)

    # write out the MS file
    write_MS(o_dict, outfile=out_MS+'_EB'+str(EB)+'.ms', direct_file=True)

    # build the file list
    o_MSlist += [out_MS+'_EB'+str(EB)+'.ms']


# concatenate the regridded files to a single MS
os.system('rm -rf '+out_MS+'.ms*')
concat(vis=o_MSlist, concatvis=out_MS+'.ms', dirtol='0.1arcsec', freqtol='0Hz')

# clean up extraneous files
[os.system('rm -rf '+in_MS+'_EB'+str(j)+'.ms') for j in range(i_dict['Nobs'])]
[os.system('rm -rf '+out_MS+'_EB'+str(j)+'.ms') for j in range(i_dict['Nobs'])]
