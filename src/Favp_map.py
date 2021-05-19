"""Produces Favg_map to be used for further analysis"""

import numpy as np
import healpy as hp
from astropy.io import fits
import pyccl as ccl

import sys
sys.path.insert(1, '../src/')
from lib import *

#parameters
NSELFUNC = 1000 #number of selection functions
tol = 0.8 #threshold for 1/F mean calculation

#read in random for mask generation
dr_elg_ran = np.load("../dat/elg_ran1024.npy")
mask = np.copy(dr_elg_ran)
mask[dr_elg_ran != 0] = 1 #good pixels are 1
mask = mask.astype("bool")

#store values for diagnostics
Fmean_pre = np.zeros(nselfunc) #mean of sel funcs prior norm correction
Fmean_post = np.zeros(nselfunc) #mean of sel funcs post norm correction
invFmean_pre = np.zeros(nselfunc) #mean of sel funcs prior norm correction
invFmean_post = np.zeros(nselfunc) #mean of sel funcs post norm correction
Favg_map = np.zeros(12*NSIDE**2) #average selection function

for i in range(NSELFUNC):
    tmpF = read_img_map(flist[i])
    masked_tmpF_mean = np.mean(tmpF[mask > 0]) #mean of sel. func. tmpF
    Fmean_pre[i] = np.mean(tmpF[mask > 0]) #means of sel. funcs
    invFmean_pre[i] = np.mean(1/tmpF[tmpF > tol])

    #Renormalize map to <F> = 1; Mehdi clipped extreme values so <F> != 1
    tmpF = tmpF/masked_tmpF_mean
    Fmean_post[i] = np.mean(tmpF[mask > 0])
    invFmean_post[i] = np.mean(1/tmpF[tmpF > tol])

    #define running average
    Favg_map = (tmpF + i * Favg_map)/(i + 1)

    if((i % (NSELFUNC//10)) == 0):
        print(i)

print('Pre normalization')
print('------------')
print(f'Fmean -- mean: {np.mean(Fmean_pre)}, low: {np.min(Fmean_pre)}, high: {np.max(Fmean_pre)}')
print(f'invFmean -- mean: {np.mean(invFmean_pre)}, low: {np.min(invFmean_pre)}, high: {np.max(invFmean_pre)}')
print('------------')
print('Post normalization')
print('------------')
print(f'Fmean -- mean: {np.mean(Fmean_post)}, low: {np.min(Fmean_post)}, high: {np.max(Fmean_post)}')
#print(f'Favg_map -- mean: {np.mean(Favg_map[mask>0])}, low: {np.min(Favg_map[mask>0])}, high: {np.max(Favg_map[mask>0])}')
print(f'invFmean -- mean: {np.mean(invFmean_post)}, low: {np.min(invFmean_post)}, high: {np.max(invFmean_post)}')

np.save("../dat/windows_1000mocks/Favg/Favg_map.npy", Favg_map)
