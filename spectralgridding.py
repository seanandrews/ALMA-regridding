import numpy as np
import cv2
from scipy.interpolate import interp1d

def fzp_sample(data, nsample):
    """
    FFT zero-padding interpolation: 
    adding zeros at higher frequencies of Fourier coefficient
    """
    nch,ndraws = data.shape[0], data.shape[1]
    # nch has to be an even number for zeropadding below
    if (nch % 2 == 1):
        data = data[:-1,]
        nch -= 1
        
    # fft    
    fdata = np.fft.fft(data, axis=0)
    # zero-padding
    fpad = np.zeros([nsample, ndraws], dtype = complex)
    h = nch//2
    fpad[0:h, :] = fdata[0:h, :]
    fpad[nsample-h+1:, :] = fdata[h+1:, :]
    # ifft
    fsample = np.fft.ifft(fpad, axis=0)*nsample/nch
    
    return fsample
    
    
def fftshift(xs, vals, nu_shift):
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
    

def lanczos_sample(data, nsample):
    """
    Lanczos interpolation using OpenCV libary
    """
    img = np.array([data])
    resample = cv2.resize(img, (int(nsample*(1+1/len(data))), 1),
                          interpolation=cv2.INTER_LANCZOS4)[0]
    # Pick the central range due to extra pixels at each side
    ys = resample[int(nsample/(2*len(data))):int(nsample/(2*len(data)))+nsample]    
    return ys
    
    