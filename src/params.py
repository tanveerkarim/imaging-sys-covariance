import numpy as np

#plotting params
fs = 20
fsize = (10, 7)

#mock definitions
tol = 0.8
NMOCKS = 1000

#Fourier Space definitions
NSIDE = 1024
LMIN = 100
LMAX = 2 * NSIDE
ell = np.arange(0, LMAX, 1)
ELL = np.logspace(0, np.log10(LMAX), 10)

#cosmology in CLASS formalism
h = 0.6736
Omega_c = 0.12/h**2
Omega_b = 0.02237/h**2
A_s = 2.083e-09
n_s = 0.9649
b1 = 1.75

#fsky
mask = np.load("../dat/elg_ran1024.npy")
mask[mask != 0] = 1 #good pixels are 1
mask = mask.astype("bool")
fsky = np.sum(mask)/mask.size