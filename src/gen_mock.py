"""This script generates delta_g and noise_g realizations. The same realizations
will be used for all the 6 experiments. This code is parallelized using 
Multiprocessing."""

import numpy as np
import healpy as hp

import sys
sys.path.insert(1, '/home/tkarim/imaging-sys-covariance/src/')
import params as pm
from lib import *

from time import time 

from multiprocessing import Pool

start_time = time()
#import theory signal Cls generated from CLASS
cls_elg_th = np.load("../dat/cosmology_ini/gaussian_mocks/cl_th.npy")

#generate theory noise Cls 
nbar_sr = (np.pi/180)**(-2) * pm.nbar_sqdeg #conversion factor from sq deg to sr
cls_shot_noise = 1/nbar_sr * np.ones_like(pm.ell)

#define seed
SEED = 42

#define multiprocessing parameters
nprocesses = 40
niter = int(pm.NMOCKS/nprocesses) #number of times to loop over Pool
j = np.arange(nprocesses) #idx for simultaneous number of parallel calculations 

#output dir
mock_dir = "/mnt/gosling1/tkarim/img-sys/mocks/"

def genMock(seed):
    rng = np.random.default_rng(seed) #set random number generator

    map_signal = hp.synfast(cls = cls_elg_th, nside = pm.NSIDE, pol = False, 
                verbose = False)
    map_noise = hp.synfast(cls = cls_shot_noise, nside = pm.NSIDE, pol = False,
                verbose = False)

    return map_signal, map_noise

#launch parallel processes
for i in range(niter):
    with Pool(nprocesses) as p:
        idx = i * nprocesses + j
        signal_and_noise_maps = np.array(p.map(genMock, idx + SEED))
        maps_signal = signal_and_noise_maps[:,0,:]
        maps_noise = signal_and_noise_maps[:,1,:]

        for k, mapIdx in enumerate(idx):
            np.save(mock_dir + f"signal/signalMap_{mapIdx}.npy", maps_signal[k], allow_pickle=False)
            np.save(mock_dir + f"noise/noiseMap_{mapIdx}.npy", maps_noise[k], allow_pickle=False)

end_time = time()

print(f"Total time taken: {end_time - start_time}")