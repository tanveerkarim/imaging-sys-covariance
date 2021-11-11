"""
This script generates gaussian mocks on the fly and calculates pCls.This is
a parellized code to be run on jobs array to launch multiple experiments 
simultaneously.
"""

import os 
os.environ["OMP_NUM_THREADS"] = "16"

import numpy as np
import healpy as hp
from multiprocessing import Pool
import pickle 

import sys
sys.path.insert(1, '/home/tkarim/imaging-sys-covariance/src/')
import params as pm
from lib import read_img_map

import argparse

from time import time 

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
if(window_type == 'linear'): #linear
    flist = np.load("../dat/flist_window_linear.npy")
    map_sys_avg = np.load("../dat/sysavg_map_lin.npy") #read in the average map 
    outp_dir = data_dir + "stats/linear/"
elif(window_type == 'nn'): #neural network
    flist = np.load("../dat/flist_window_nn.npy")
    map_sys_avg = np.load("../dat/sysavg_map_nn.npy")  # read in the average map
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
mask_base = np.load("../dat/mask_bool_dr9.npy")

##FUNCTIONS


# TO DO: HAVE TO DEFINE MASKS APPROPRIATELY
def contaminate_map(map_signal, map_noise, map_sys, map_mask, expname):
    """
    Returns contaminated maps based on Karim et al. 2021 definitions
    
        Parameters:
            map_signal (healpy map): map of delta map_signal
            map_noise  (healpy map): map of delta map_noise 
            map_sys    (healpy map): map of imaging systematics
            map_mask   (healpy map): boolean map of footprint; pix_good == 1 
            expname (str) : definition of delta contaminate map 
        
        Returns:
            map_cont (healpy map) : map of contaminated map_signal
    """

    map_cont = np.zeros(map_signal.shape[0])
    map_cont[:] = hp.UNSEEN  # default set to UNSEEN

    #renormalize window such that <W_g> = 1.
    masked_window_mean = np.mean(map_sys[map_mask > 0])
    map_sys = map_sys/masked_window_mean

    if(expname == 'A'):
        map_sys_known = map_sys_avg  # idealized known window
        map_sys_known /= masked_window_mean

        map_cont[map_mask] = map_sys_known[map_mask]*map_signal[map_mask] + \
            np.sqrt(map_sys_known[map_mask])*map_noise[map_mask]
    elif(expname == 'B'):
        map_cont[map_mask] = map_sys[map_mask]*map_signal[map_mask] + \
            np.sqrt(map_sys[map_mask])*map_noise[map_mask]
    elif(expname == 'C'):
        map_sys_estimated = map_sys_avg  # best estimator of window
        map_cont[map_mask] = map_sys[map_mask]*(1 + map_signal[map_mask]) + \
            np.sqrt(map_sys[map_mask])*map_noise[map_mask] - \
            map_sys_estimated[map_mask]
    elif(expname == 'D'):
        map_sys_known = map_sys_avg
        map_sys_known /= masked_window_mean

        map_cont[map_mask] = map_signal[map_mask] + \
            np.sqrt(1/map_sys_known[map_mask])*map_noise[map_mask]
    elif(expname == 'E'):
        map_sys_estimated = map_sys_avg
        map_sys_estimated /= masked_window_mean

        map_cont[map_mask] = (map_sys[map_mask]*map_signal[map_mask])/map_sys_estimated[map_mask] + \
            (np.sqrt(map_sys[map_mask])*map_noise[map_mask]) / \
            map_sys_estimated[map_mask]
    elif(expname == 'F'):
        map_sys_estimated = map_sys_avg
        map_sys_estimated /= masked_window_mean

        map_cont[map_mask] = (map_sys[map_mask]*(1. + \
            map_signal[map_mask]))/map_sys_estimated[map_mask] + \
            (np.sqrt(map_sys[map_mask])*map_noise[map_mask]) / \
            map_sys_estimated[map_mask] - 1
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
        elif((experiment == 'D') | (experiment == 'E') | (experiment == 'F')):
            map_mask = mask_base & (map_sys_avg > pm.tol) #need to overmask 
                                                        #since 1/map_sys_avg
        #elif((experiment == 'E') | (experiment == 'F')): deprecate to use same mask
        #    map_mask = mask_base & (map_sys > pm.tol)

        #contaminate map per experiment
        map_contaminated = contaminate_map(map_signal = map_signal,
                            map_noise = map_noise, map_sys = map_sys,
                            map_mask = map_mask,
                            expname = experiment)
    
        #calcuate pseudo-Cl
        pcls_idx[experiment] = hp.anafast(map_contaminated, lmax=pm.LMAX - 1, 
                                            pol=False)
        #calculate fsky
        #fsky_idx[experiment] = np.mean(map_mask)
        #calculate window noise
        #if(experiment == 'A'):
        #    n_window_idx[experiment] = np.mean(map_sys_avg[map_mask]) * 1/nbar_sr
        #elif(experiment == 'D'):
        #    n_window_idx[experiment] = np.mean(1/map_sys_avg[map_mask]) * 1/nbar_sr
        #if((experiment == 'B') | (experiment == 'C')):
        #    n_window_idx[experiment] = np.mean(map_sys[map_mask]) * 1/nbar_sr
        #elif((experiment == 'E') | (experiment == 'F')):
        #    n_window_idx[experiment] = np.mean(1/map_sys[map_mask]) * 1/nbar_sr
        #n_window_idx[experiment] = np.mean(1/map_sys[map_mask]) * 1/nbar_sr

    return pcls_idx#, fsky_idx, n_window_idx

pcls_dict = {}#; fsky_dict = {}; n_window_dict = {} #for storing values

start_time = time()

for i in range(nmaps//njobs_parallel): #i refers to chunk of maps processed together
    idx = JOB_ID*int(100) + i * njobs_parallel + np.arange(njobs_parallel)
    with Pool(njobs_parallel) as p:
        output = p.map(getpCls, idx)
    pcls_dict[i] = output[0]
    #fsky_dict[i] = output[1]
    #n_window_dict[i] = output[2]

stats = {'pcls': pcls_dict}#, 'fsky': fsky_dict, 'window_noise': n_window_dict}

end_time = time()
print(f"100 jobs took {end_time - start_time} seconds.")

pickle.dump(stats, open(outp_dir + str(JOB_ID) + ".npy", "wb"))

#np.save(, stats)