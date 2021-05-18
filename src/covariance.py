import numpy as np
import pandas as pd
import healpy as hp
from astropy.io import fits
import pyccl as ccl

from tqdm import tqdm #for timing

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-notebook")

import sys
sys.path.insert(1, '/home/tanveer/Documents/desi-planck-cross-corr/imaging-sys-covariance/src/')
from lib import *

#plotting parameters
fs = 20 #font size
fsize = (10, 7) #figure size

#cosmology and simulation parameters
NSIDE = 1024
LMIN = 100;
LMAX = 3 * NSIDE - 1
ell = np.arange(0, LMAX, 1)
ELL = np.logspace(0, np.log10(LMAX), 10) #for binning

h = 0.6736
Omega_c = 0.12/h**2
Omega_b = 0.02237/h**2
A_s = 2.083e-09
n_s = 0.9649

NMOCKS = 100
