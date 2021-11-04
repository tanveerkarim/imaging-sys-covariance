"""
This script generates gaussian mocks on the fly and calculates pCls.This is
a parellized code to be run on jobs array to launch multiple experiments 
simultaneously.
"""

import os 
os.environ["OMP_NUM_THREADS"] = 16

import numpy as np
import healpy as hp
from multiprocessing import Pool

import sys
sys.path.insert(1, '/home/tkarim/imaging-sys-covariance/src/')
import params as pm
from lib import read_img_map

import argparse

#from time import time 

#parser for bash arguments
parser = argparse.ArgumentParser()
parser.add_argument("--JOB_ID", "-jid", type = int, help = "")
parser.add_argument("--window_type", "-wt", type = str, help = "lin -> linear; nn -> neural network")
args = parser.parse_args()
JOB_ID = args.JOB_ID
window_type = args.window_type

##GLOBALS
nmaps = pm.NMOCKS #number of maps in this single node
njobs_parallel = 5 #number of parallel processes in this node
data_dir = "/mnt/gosling1/tkarim/img-sys/"
experiments = ['A', 'B', 'C', 'D', 'E', 'F'] #list of definitions 

#list of all selection function fits files
if(window_type == 'lin'): #linear
    flist = np.load("../dat/flist_window_linear.npy")
    map_sys_avg = np.load("") #read in the average map 
    outp_dir = data_dir + "stats/linear/"
elif(window_type == 'nn'): #neural network
    flist = np.load("../dat/flist_window_nn.npy")
    map_sys_avg = np.load("")  # read in the average map
    outp_dir = data_dir + "stats/nn/"
else:
    raise ValueError("Wrong window type entered.")

#import theory signal Cls generated from CLASS
cls_elg_th = np.load("../dat/cosmology_ini/gaussian_mocks/cl_th.npy")

#generate theory noise Cls 
nbar_sr = (np.pi/180)**(-2) * pm.nbar_sqdeg #conversion factor from sq deg to sr
cls_shot_noise = 1/nbar_sr * np.ones_like(pm.ell)

#define seed
SEED = 42

#import base mask; defined by fpix on DR9 ELG randoms
mask_base = np.load("")

##FUNCTIONS
def contaminate_map(signal, noise, sys, mask, expname): ##TO DO: HAVE TO DEFINE MASKS APPROPRIATELY
    """
    Returns contaminated maps based on Karim et al. 2021 definitions
    
        Parameters:
            signal (healpy map): map of delta signal
            noise  (healpy map): map of delta noise 
            sys    (healpy map): map of imaging systematics
            mask   (healpy map): boolean map of footprint; pix_good == 1 
            expname (str) : definition of delta contaminate map 
        
        Returns:
            map_cont (healpy map) : map of contaminated signal
    """
    
    map_cont = np.zeros(signal.shape[0])
    map_cont[:] = hp.UNSEEN  # default set to UNSEEN
    
    if(expname == 'A'):
        sys_known = map_sys_avg  # idealized known window
        map_cont[mask] = sys_known[mask]*signal[mask] + \
            np.sqrt(sys_known[mask])*noise[mask]
    elif(expname == 'B'):
        map_cont[mask] = sys[mask]*signal[mask] + \
            np.sqrt(sys[mask])*noise[mask]
    elif(expname == 'C'):
        sys_estimated = map_sys_avg #best estimator of window
        map_cont[mask] = sys[mask]*(1 + signal[mask]) + \
            np.sqrt(sys[mask])*noise[mask] - sys_estimated[mask]
    elif(expname == 'D'):
        sys_known = map_sys_avg
        map_cont[mask] = signal[mask] + np.sqrt(1/sys_known[mask])*noise[mask]
    elif(expname == 'E'):
        sys_estimated = map_sys_avg
        map_cont[mask] = (sys[mask]*signal[mask])/sys_estimated[mask] + \
            (np.sqrt(sys[mask])*noise[mask])/sys_estimated[mask]
    elif(expname == 'F'):
        sys_estimated = map_sys_avg
        map_cont[mask] = (sys[mask]*(1. + signal[mask]))/sys_estimated[mask] +
        (np.sqrt(sys[mask])*noise[mask])/sys_estimated[mask] - 1
    else:
        raise ValueError("Wrong experiment name entered.")

    return map_cont 

def genMock(idx):  # apply window at this
    """
    Returns signal, noise and systematics map given index
    """
    
    rng_signal = np.random.default_rng(SEED + idx)  # set random number generator
    map_signal = hp.synfast(cls=cls_elg_th, nside=pm.NSIDE, pol=False,
                            verbose=False, rng=rng_signal)
    
    rng_noise  = np.random.default_rng(2000 + SEED + idx)
    map_noise = hp.synfast(cls=cls_shot_noise, nside=pm.NSIDE, pol=False,
                           verbose=False, rng=rng_noise)
    
    map_sys = read_img_map(flist[idx])

    return map_signal, map_noise, map_sys

def getpCls(idx):
    """
    Returns pcl, fsky and n_window given index
    """
    
    map_signal, map_noise, map_sys = genMock(idx)
    
    pcls_idx = {}; fsky_idx = {}; n_window_idx = {} #for storing loop values

    for experiment in experiments: #loop over all definitions
        ##--MAKE MASK FOR GIVEN EXPERIMENT
        
        if((experiment == 'A') | (experiment == 'B') | (experiment == 'C')):
            map_mask = mask_base
        elif(experiment == 'D'):
            map_mask = mask_base & (map_sys_avg > pm.tol) #need to overmask 
                                                        #since 1/map_sys_avg
        elif((experiment == 'E') | (experiment == 'F')):
            map_mask = mask_base & (map_sys > pm.tol)

        #renormalize window such that <W_g> = 1.
        masked_window_mean = np.mean(map_sys[map_mask > 0])
        map_sys = map_sys/masked_window_mean

        #contaminate map per experiment
        map_contaminated = contaminate_map(signal = map_signal,
                            noise = map_noise, sys = map_sys,
                            mask = map_mask,
                            expname = experiment)
    
        #calcuate pseudo-Cl
        pcls_idx[experiment] = hp.anafast(map_contaminated, lmax=pm.LMAX - 1, 
                                            pol=False)
        #calculate fsky
        fsky_idx[experiment] = np.mean(map_mask)
        #calculate window noise
        if(experiment == 'A'):
            n_window_idx[experiment] = np.mean(map_sys_avg[mask]) * 1/nbar_sr
        elif(experiment == 'D'):
            n_window_idx[experiment] = np.mean(1/map_sys_avg[map_mask]) * 1/nbar_sr
        if((experiment == 'B') | (experiment == 'C'))::
            n_window_idx[experiment] = np.mean(map_sys[mask]) * 1/nbar_sr
        elif((experiment == 'E') | (experiment == 'F')):
            n_window_idx[experiment] = np.mean(1/map_sys[map_mask]) * 1/nbar_sr
        n_window_idx[experiment] = 

    return pcls_idx, fsky_idx, n_window_idx

pcls_dict = {}; fsky_dict = {}; n_window_dict = {} #for storing values

 for i in range(nmaps//njobs_parallel): #i refers to chunk of maps processed together
    idx = JOB_ID + i * njobs_parallel + np.arange(njobs_parallel)
    with Pool(njobs_parallel):
        output = Pool.map(getpCls, idx)
    pcls_dict[i] = output[0]
    fsky_dict[i] = output[1]
    n_window_dict[i] = output[2]

stats = {'pcls': pcls_dict, 'fsky': fsky_dict, 'window_noise': n_window_dict}

np.save(outp_dir + JOB_ID + ".npy", stats)