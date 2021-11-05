import numpy as np

#plotting params
fs = 20
fsize = (10, 7)

#mock definitions
tol = 0.8
NMOCKS = 1000
nbar_sqdeg = 2400 #number density of galaxies

#Fourier Space definitions
NSIDE = 1024
LMIN = 100
LMAX = NSIDE
ell = np.arange(0, LMAX, 1)
ELL = np.logspace(0, np.log10(LMAX), 10)

#fsky
mask = np.load("../dat/mask_bool_dr9.npy")
fsky = np.sum(mask)/mask.size

#linear bias
b1 = 1.75