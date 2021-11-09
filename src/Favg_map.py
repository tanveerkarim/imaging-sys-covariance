"""Produces sysavg_map to be used for further analysis"""

import numpy as np
import healpy as hp

import sys
sys.path.insert(1, '../src/')
from lib import *
import params as pm

#list of all selection function fits files
flist = np.load("../dat/flist_window_linear.npy")

#read in random for mask generation
mask = np.load("../dat/mask_bool_dr9.npy")

#store values for diagnostics
sysmean_pre = np.zeros(pm.NMOCKS) #mean of sys map prior norm correction
sysmean_post = np.zeros(pm.NMOCKS)  # mean of sys map post norm correction
invsysmean_pre = np.zeros(pm.NMOCKS)  # mean of sys map prior norm correction
invsysmean_post = np.zeros(pm.NMOCKS) #mean of sys map post norm correction
sysavg_map = np.zeros(12*pm.NSIDE**2)  # average sys map

for i in range(len(flist)):
    tmpsys = read_img_map(flist[i])
    masked_tmpsys_mean = np.mean(tmpsys[mask > 0]) #mean of sys map tmpsys
    sysmean_pre[i] = np.mean(tmpsys[mask > 0])  # means of sys map
    invsysmean_pre[i] = np.mean(1/tmpsys[tmpsys > pm.tol])

    #Renormalize map to <sys> = 1; Mehdi clipped extreme values so <sys> != 1
    tmpsys = tmpsys/masked_tmpsys_mean
    sysmean_post[i] = np.mean(tmpsys[mask > 0])
    invsysmean_post[i] = np.mean(1/tmpsys[tmpsys > pm.tol])

    #define running average
    sysavg_map = (tmpsys + i * sysavg_map)/(i + 1)

    if((i % (pm.NMOCKS//10)) == 0):
        print(i)

print('Pre normalization')
print('------------')
print(f'Sys map mean -- mean: {np.mean(sysmean_pre)}, low: {np.min(sysmean_pre)}, high: {np.max(sysmean_pre)}')
print(f'invsysmap mean -- mean: {np.mean(invsysmean_pre)}, low: {np.min(invsysmean_pre)}, high: {np.max(invsysmean_pre)}')
print('------------')
print('Post normalization')
print('------------')
print(f'Sys map mean -- mean: {np.mean(sysmean_post)}, low: {np.min(sysmean_post)}, high: {np.max(sysmean_post)}')
#print(f'sysavg_map -- mean: {np.mean(sysavg_map[mask>0])}, low: {np.min(sysavg_map[mask>0])}, high: {np.max(sysavg_map[mask>0])}')
print(f'invsysmap mean -- mean: {np.mean(invsysmean_post)}, low: {np.min(invsysmean_post)}, high: {np.max(invsysmean_post)}')

np.save("../dat/sysavg_map_lin.npy", sysavg_map)
