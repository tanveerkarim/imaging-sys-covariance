"""Calculate pCls based on mocks"""

import numpy as np
import pandas as pd
import healpy as hp
from astropy.io import fits
import pyccl as ccl

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-notebook")

import sys
sys.path.insert(1, '/home/tkarim/imaging-sys-covariance/src/')
from lib import *
import params as pm

#list of all selection function fits files
import glob
flist = glob.glob("/home/tkarim/imaging-sys-covariance/dat/windows/*fits")

#plotting parameters
fs = 20 #font size
fsize = (10, 7) #figure size

#noise parameters
nbar_sqdeg = 2400 #per deg2
nbar_sr = (np.pi/180)**(-2) * nbar_sqdeg #conversion factor from sq deg to sr
cls_shot_noise = 1/nbar_sr * np.ones_like(pm.ell)

#read in random and degrade it to generate mask
dr_elg_ran = np.load("../dat/elg_ran1024.npy")
#dr_elg_ran64 = hp.ud_grade(dr_elg_ran, 64) #make it very coarse to remove weird effects in the middle
#dr_elg_ran_final = hp.ud_grade(dr_elg_ran64, pm.NSIDE) #leave as be for stellar masks

mask = np.copy(dr_elg_ran)
mask[dr_elg_ran != 0] = 1 #good pixels are 1
mask = mask.astype("bool")

#import avg map
Favg_map = np.load("../dat/windows/Favg/Favg_map_unpickled.npy")

#set theory Cls
cls_elg_th = cgll(ell = pm.ell, bias = pm.b1, Omega_c = pm.Omega_c,
                    Omega_b = pm.Omega_b, h = pm.h, A_s = pm.A_s, 
                    n_s = pm.n_s)

##MAIN PART OF THE CODE##

#print("Which model do you want to calculate?")
#expname = input() #Look at notebook for definitions
expname = sys.argv[1]
print("Running Model " + expname)

cls_obs = np.zeros((pm.NMOCKS, pm.LMAX)) #pCl values
nl = np.zeros(pm.NMOCKS) #selection function noise
fsky = np.zeros(pm.NMOCKS)

#set const. fsky and nl; fsky same in the first three models
if((expname == 'A')):
    fsky = np.sum(mask)/mask.shape[0] * np.ones(pm.NMOCKS)
    nl = np.mean(Favg_map)*1/nbar_sr*np.ones(pm.NMOCKS)
    tmpF = Favg_map #since fixed window set outside loop
    additive = None
elif((expname == 'B') | (expname == 'C')):
    fsky = np.sum(mask)/mask.shape[0] * np.ones(pm.NMOCKS)
elif(expname == 'D'):
    mask = mask & (Favg_map > pm.tol)
    fsky = np.sum(mask)/mask.shape[0] * np.ones(pm.NMOCKS)
    nl = np.mean(1/Favg_map[mask]) * 1/nbar_sr * np.ones(pm.NMOCKS)
    tmpF = Favg_map #since fixed window set outside loop

#model conditions; window applied to data vs random
if((expname == 'A') | (expname == 'B') | (expname == 'C')):
    img_applied_data = False
else:
    img_applied_data = True

#model conditions; additive component
if((expname == 'C') | (expname == 'F')):
    additive = Favg_map
else:
    additive = None

#loop over mocks to calculate pCl
for i in range(pm.NMOCKS):

    #read in sel. function
    if((expname != 'A') & (expname != 'D')):
        tmpF = read_img_map(flist[i])

        #renormalize selection function such that <F> = 1.
        masked_tmpF_mean = np.mean(tmpF[mask > 0])
        tmpF /= masked_tmpF_mean

    #set mask for E and F
    if((expname == 'E') | (expname == 'F')):
        mask = mask & (tmpF > pm.tol)
        fsky[i] = np.sum(mask)/mask.shape[0]
        nl[i] = np.mean(1/tmpF[mask]) * 1/nbar_sr
    elif((expname == 'B') | (expname == 'C')):
        nl[i] = np.mean(tmpF)*1/nbar_sr

    #calculate pCl
    cls_obs[i] = cls_from_mock(cls_th = cls_elg_th,
                cls_shot_noise=cls_shot_noise, F = tmpF,
                mask = mask, seed = 67 + 2*i, LMAX = pm.LMAX, additive = additive,
                img_applied_data = img_applied_data)

    if((i % (100)) == 0):
        print(i)

#first order correction to pCls; fsky and nl
if((expname == 'A') | (expname == 'B') | (expname == 'C')):
    cls_obs = (cls_obs - nl[:,np.newaxis])/fsky[:,np.newaxis]
else:
    cls_obs = cls_obs/fsky[:,np.newaxis] - nl[:,np.newaxis]

#store values
np.save("../dat/pCls/1000mocks/pCls_" + expname + ".npy", cls_obs)
np.save("../dat/pCls/1000mocks/noise_window_" + expname + ".npy", nl)
np.save("../dat/pCls/1000mocks/fsky_" + expname + ".npy", fsky)