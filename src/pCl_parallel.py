"""This script generates gaussian mocks on the fly and calculates pCls.This is
 a parellized code to be run on jobs array to launch multiple experiments 
 simultaneously."""

import os 
os.environ["OMP_NUM_THREADS"] = 16

import numpy as np
import healpy as hp
from multiprocessing import Pool

import sys
sys.path.insert(1, '/home/tkarim/imaging-sys-covariance/src/')
import params as pm
from lib import *

from time import time 

JOB_ID = sys.argv[1]

##GLOBALS
nmaps = 1000 #number of maps in this single node
njobs_parallel = 5 #number of parallel processes in this node

experiments = ['A'.. 'F']

#import theory signal Cls generated from CLASS
cls_elg_th = np.load("../dat/cosmology_ini/gaussian_mocks/cl_th.npy")

#generate theory noise Cls 
nbar_sr = (np.pi/180)**(-2) * pm.nbar_sqdeg #conversion factor from sq deg to sr
cls_shot_noise = 1/nbar_sr * np.ones_like(pm.ell)

#define seed
SEED = 42

#output dir
outp_dir = "/mnt/gosling1/tkarim/img-sys/stats/"

##FUNCTIONS
def genMock(seed):  # apply window at this
    rng = np.random.default_rng(seed)  # set random number generator

    map_signal = hp.synfast(cls=cls_elg_th, nside=pm.NSIDE, pol=False,
                            verbose=False)
    map_noise = hp.synfast(cls=cls_shot_noise, nside=pm.NSIDE, pol=False,
                           verbose=False)

    return map_signal, map_noise

getpCls(idx):
    map_signal, map_noise = genMock(idx)
    
    """DO calculation """
    for experiment in experiments: 

    return pcls, fsky, n_window

pcls = {}; fsky = {}; n_window = {}

 for i in range(nmaps//njobs_parallel):
    idx = i * njobs_parallel + np.arange(njobs_parallel)
    with Pool(njobs_parallel):
        output = Pool.map(getpCls, idx)
    pcls[i] = output[0]
    fsky[i] = output[1]
    n_window = output[2]

np.save(values)
