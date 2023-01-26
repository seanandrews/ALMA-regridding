import numpy as np
from scipy.interpolate import interp1d


def interp_nearest(x_in, y_in, x_out, axis=0):

    # interpolation function
    fint = interp1d(x_in, y_in, axis=axis, kind='nearest')

    # return interpolates
    return fint(x_out)


def interp_linear(x_in, y_in, x_out, axis=0):

    # interpolation function
    fint = interp1d(x_in, y_in, axis=axis, kind='linear')

    # return interpolates
    return fint(x_out)


def interp_cubic(x_in, y_in, x_out, axis=0):

    # interpolation function
    fint = interp1d(x_in, y_in, axis=axis, kind='cubic')

    # return interpolates
    return fint(x_out)


def interp_fftshift(xs, vals, nu_shift):
    """
    Shift the function by applying a phase shift in the frequency domain

    Args:
        nu_shift: how much to shift by, in units of Hz
    """
    nch, ndraws = vals.shape[0], vals.shape[1]
    if (nch % 2 == 1):
        vals = vals[:-1,]
        xs = xs[:-1]
        nch -= 1    
        
    assert nch % 2 == 0, "Only even numbered arrays for now."
    dchan = xs[1] - xs[0]

    # use fft to access cross-correlation function
    rho_packed = np.fft.fft(np.fft.fftshift(vals,axes=0),axis=0)
    fs_packed = np.fft.fftfreq(nch, d=dchan)

    # mulitply rho_packed by phase shift 
    # if shifting by positive dnu, then 
    phase = np.exp(-2.0j * np.pi * fs_packed * nu_shift)
    phase2 = np.tile(phase,(ndraws,1)).T
    
    # transform back using ifft
    rho_packed_shifted = rho_packed * phase2
    return np.fft.fftshift(np.fft.ifft(rho_packed_shifted,axis=0),axes=0)
