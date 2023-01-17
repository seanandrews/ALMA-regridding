#!/usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import spectralgridding as sg
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from matplotlib import font_manager
plt.style.use(['default', 'nice_line.mplstyle'])
plt.rcParams.update({'font.family':'sans-serif'})

# Define a snippet of channels and upsampled sub-channels
nch, upsample, nedge = 22, 20, 5
ch = np.linspace(0, 21, nch)
ch_ = np.interp(np.arange((nch - 1) * upsample + 1),
                np.arange(0, nch * upsample, upsample), ch)
if (len(ch_) % 2 == 1):
    ch_ = ch_[:-1]

# Calculate random draws from a Gaussian covariance matrix
ndraws = 1000000
raw_draws = np.random.normal(0, 1, (nch, ndraws))

# Convolve with the SRF
wsu_draws = 1. * raw_draws	# effectively no SRF for WSU (?)
xch = ch - np.mean(ch)
now_SRF = 0.5*np.sinc(xch) + 0.25*np.sinc(xch-1) + 0.25*np.sinc(xch+1)
now_draws = convolve1d(np.sqrt(8./3.)*raw_draws, now_SRF/np.sum(now_SRF), 
                       axis=0, mode='nearest')
bin_draws = np.average(now_draws.reshape(-1, 2, ndraws), axis=1)

# Nearest neighbor interpolation of the draws; "noise" of the interpolates
nn_ = interp1d(ch, wsu_draws, kind='nearest', axis=0)
nn_noise = np.std(nn_(ch_), axis=1)
nnn_ = interp1d(ch, now_draws, kind='nearest', axis=0)
nnn_noise = np.std(nnn_(ch_), axis=1)
bch = np.average(ch.reshape(-1, 2), axis=1)
bch_ = np.average(ch_.reshape(-1, 2), axis=1)
bnn_ = interp1d(bch, bin_draws, kind='nearest', axis=0,
                fill_value='extrapolate')
bnn_noise = np.std(bnn_(bch_), axis=1)

# Linear interpolation of the draws; "noise" of the interpolates
lin_ = interp1d(ch, wsu_draws, kind='linear', axis=0)
lin_noise = np.std(lin_(ch_), axis=1)
nlin_ = interp1d(ch, now_draws, kind='linear', axis=0)
nlin_noise = np.std(nlin_(ch_), axis=1)
blin_ = interp1d(bch, bin_draws, kind='linear', axis=0, 
                 fill_value='extrapolate')
blin_noise = np.std(blin_(bch_), axis=1)

# Cubic interpolation of the draws; "noise" of the interpolates
cub_ = interp1d(ch, wsu_draws, kind='cubic', axis=0)
cub_noise = np.std(cub_(ch_), axis=1)
ncub_ = interp1d(ch, now_draws, kind='cubic', axis=0)
ncub_noise = np.std(ncub_(ch_), axis=1)
bcub_ = interp1d(bch, bin_draws, kind='cubic', axis=0,
                 fill_value='extrapolate')
bcub_noise = np.std(bcub_(bch_), axis=1)

# FFT zero-padding interpolation of the draws
fzp_ = sg.fzp_sample(wsu_draws, nch*upsample)
fzp_noise = np.std(fzp_, axis=1)
fzpn_ = sg.fzp_sample(now_draws, nch*upsample)
fzpn_noise = np.std(fzpn_, axis=1)
bfzp_ = sg.fzp_sample(bin_draws, int(nch*upsample/2))
bfzp_noise = np.std(bfzp_, axis=1)    

# FFTshift interpolation of the draws
ylist=[]
dch=ch[1]-ch[0]
for i in range(upsample):
    chshift=-1.*i*dch/upsample
    tmpdata = sg.fftshift(ch,wsu_draws,chshift)
    ylist.append(tmpdata)
    
fsh_ = (np.array(ylist).T).reshape(ndraws,-1)
fsh_noise = np.std(fsh_, axis=0)

ylist=[]
for i in range(upsample):
    chshift=-1.*i*dch/upsample
    tmpdata = sg.fftshift(ch,now_draws,chshift)
    ylist.append(tmpdata)
    
fshn_ = (np.array(ylist).T).reshape(ndraws,-1)
fshn_noise = np.std(fshn_, axis=0)

ylist=[]
dch=bch[1]-bch[0]
for i in range(upsample):
    chshift=-1.*i*dch/upsample
    tmpdata = sg.fftshift(bch,bin_draws,chshift)
    ylist.append(tmpdata)
    
bfsh_ = (np.array(ylist).T).reshape(ndraws,-1)
bfsh_noise = np.std(bfsh_, axis=0)

# Lanczos interpolation of the draws
lan_= []
lann_ = []
blan_ =[]
for i in range(ndraws):
    y=wsu_draws[:,i]
    nsample = nch*upsample
    ys = sg.lanczos_sample(y, nsample)
    lan_.append(ys)
    
    y=now_draws[:,i]
    ys = sg.lanczos_sample(y, nsample)
    lann_.append(ys)
    
    y=bin_draws[:,i]
    nsample = int(nch*upsample/2.)
    ys = sg.lanczos_sample(y, nsample)
    blan_.append(ys)

lan_ = np.asarray(lan_)
lan_noise = np.std(lan_, axis=0)

lann_ = np.asarray(lann_)
lann_noise = np.std(lann_, axis=0)

blan_ = np.asarray(blan_)
blan_noise = np.std(blan_, axis=0)

# manually reset noise spectra size for fzp, fsh and lan
fzp_noise = fzp_noise[0:420]
fzpn_noise = fzpn_noise[0:420]
bfzp_noise = bfzp_noise[0:210]

fsh_noise = fsh_noise[0:420]
fshn_noise = fshn_noise[0:420]
tmpbch_ = bch_[0:200]

lan_noise = lan_noise[0:420]
lann_noise = lann_noise[0:420]
blan_noise = blan_noise[0:210]

# Clip edge behavior
nn_noise = nn_noise[upsample*nedge:len(ch_)-upsample*nedge]
nnn_noise = nnn_noise[upsample*nedge:len(ch_)-upsample*nedge]
bnn_noise = bnn_noise[int(upsample*nedge/2):len(bch_)-int(upsample*nedge/2)]
lin_noise = lin_noise[upsample*nedge:len(ch_)-upsample*nedge]
nlin_noise = nlin_noise[upsample*nedge:len(ch_)-upsample*nedge]
blin_noise = blin_noise[int(upsample*nedge/2):len(bch_)-int(upsample*nedge/2)]
cub_noise = cub_noise[upsample*nedge:len(ch_)-upsample*nedge]
ncub_noise = ncub_noise[upsample*nedge:len(ch_)-upsample*nedge]
bcub_noise = bcub_noise[int(upsample*nedge/2):len(bch_)-int(upsample*nedge/2)]
fzp_noise = fzp_noise[upsample*nedge:len(ch_)-upsample*nedge]
fzpn_noise = fzpn_noise[upsample*nedge:len(ch_)-upsample*nedge]
bfzp_noise = bfzp_noise[int(upsample*nedge/2):len(bch_)-int(upsample*nedge/2)]
fsh_noise = fsh_noise[upsample*nedge:len(ch_)-upsample*nedge]
fshn_noise = fshn_noise[upsample*nedge:len(ch_)-upsample*nedge]
bfsh_noise = bfsh_noise[int(upsample*nedge/2):len(tmpbch_)-int(upsample*nedge/2)]
lan_noise = lan_noise[upsample*nedge:len(ch_)-upsample*nedge]
lann_noise = lann_noise[upsample*nedge:len(ch_)-upsample*nedge]
blan_noise = blan_noise[int(upsample*nedge/2):len(tmpbch_)-int(upsample*nedge/2)]


ch_ = ch_[upsample*nedge:len(ch_)-upsample*nedge]
bch_ = bch_[int(upsample*nedge/2):len(bch_)-int(upsample*nedge/2)]
ch_ -= ch_[0]
bch_ -= bch_[0]
tmpbch_ = tmpbch_[int(upsample*nedge/2):len(tmpbch_)-int(upsample*nedge/2)]
tmpbch_ -= tmpbch_[0]
# set the consistent x axis for bfsh_noise
tmp = tmpbch_[0:100]


# Make the plot demo4
fig, axs = plt.subplots(figsize=(7.5, 4.6), ncols=3, nrows=2)

# nn
ax = axs[0,0]
ax.plot(ch_, nn_noise, 'C0')#, label='WSU')
ax.plot(ch_, nnn_noise, ':C0')#, label='current')
ax.plot(bch_, bnn_noise, '--C0')#, label='current, 2x bin')
ax.plot([10, 11], [1, 1], '-k', label='WSU')
ax.plot([10, 11], [1, 1], ':k', label='current')
ax.plot([10, 11], [1, 1], '--k', label='current, 2x bin')
ax.text(0.03, 0.93, 'nearest', transform=ax.transAxes, va='center', ha='left',
        color='C0', fontsize=10)
ax.legend(fontsize=8)

# linear
ax = axs[0,1]
ax.plot(ch_, lin_noise, 'C1')
ax.plot(ch_, nlin_noise, ':C1')
ax.plot(bch_, blin_noise, '--C1') 
ax.text(0.03, 0.93, 'linear', transform=ax.transAxes, va='center', ha='left',
        color='C1', fontsize=10)

# cubic
ax = axs[0,2]
ax.plot(ch_, cub_noise, 'C3')
ax.plot(ch_, ncub_noise, ':C3')
ax.plot(bch_, bcub_noise, '--C3')
ax.text(0.03, 0.93, 'cubic', transform=ax.transAxes, va='center', ha='left',
        color='C3', fontsize=10)

# fft zeropadding
ax = axs[1,0]
ax.plot(ch_, fzp_noise, 'C4')
ax.plot(ch_, fzpn_noise, ':C4')
ax.plot(bch_, bfzp_noise, '--C4')
ax.text(0.03, 0.93, 'fft zeropadding', transform=ax.transAxes, va='center', ha='left',
        color='C4', fontsize=10)
ax.set_xlabel('channel offset for interpolation')
ax.set_ylabel('interpolated noise spectrum')


# fftshift
ax = axs[1,1]
ax.plot(ch_, fsh_noise, 'C5')
ax.plot(ch_, fshn_noise, ':C5')
ax.plot(tmp, bfsh_noise, '--C5')
ax.text(0.03, 0.93, 'fftshift', transform=ax.transAxes, va='center', ha='left',
        color='C5', fontsize=10)

# lanczos
ax = axs[1,2]
ax.plot(ch_, lan_noise, 'C6')
ax.plot(ch_, lann_noise, ':C6')
ax.plot(tmp, blan_noise, '--C6')
ax.text(0.03, 0.93, 'Lanczos', transform=ax.transAxes, va='center', ha='left',
        color='C6', fontsize=10)


for i in range(3):
    for j in range(2):
        axs[j,i].set_xlim([0, 5])
        axs[j,i].set_ylim([0.5, 1.1])

fig.subplots_adjust(left=0.06, right=0.94, bottom=0.19, top=0.97, wspace=0.22)
fig.savefig('noise_demo4.pdf')



