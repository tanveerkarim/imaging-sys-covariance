import numpy as np
import healpy as hp
import fitsio as ft


class GenSys:
    """ Generator of Systematics

    Inputs:
        - window file in .fits
        - truth density contrast

    Outputs:
        - contaminated density contrast

    """
    def __init__(self, window_file, nside=256):
        """

        inputs:
            weight_file: str, path to a .fits file that has window function
        """

        # read the 'weight' file
        windows = ft.read(window_file)
        self.hpix = windows['hpix']
        self.pred = windows['weight']
        print('# of selection functions: ', self.pred.shape)

        self.npix = 12*nside*nside
        self.mask = np.zeros(self.npix, 'bool')
        self.mask[self.hpix] = True

    def contaminate(self, window, delta, mask, noisemap = None, additive = None,
        boss = False):
        """

        inputs:
            #ix: int, index of the window
            window: float array, selection function
            delta: float array, truth density contrast
            mask: bool array, mask for density contrast
            density : float, ELG number density, set to FDR value (per deg2)
            noise: boolean, whether to add noise to contaminated map or not
            additive : boolean, whether to have an additive window component
        """
        #window = self.fetch_window(ix)

        mask_ = mask & self.mask
        delta_cont = np.zeros(self.npix)
        delta_cont[:] = hp.UNSEEN

        if noisemap is not None:
            if additive is not None:
                if(boss):
                    delta_cont[mask_] = (1 + delta[mask_])*window[mask_]/additive[mask_] + \
                    noisemap[mask_]*np.sqrt(window[mask_]/additive[mask_]) - 1.
                else:
                    delta_cont[mask_] = (1 + delta[mask_])*window[mask_] + \
                    noisemap[mask_]*np.sqrt(window[mask_]) - additive[mask_]
            else:
                if(boss):
                    delta_cont[mask_] = delta[mask_] + \
                    noisemap[mask_]*np.sqrt(1/window[mask_])
                else:
                    delta_cont[mask_] = delta[mask_]*window[mask_] + \
                    noisemap[mask_]*np.sqrt(window[mask_])
        else:
            delta_cont[mask_] = delta[mask_]*window[mask_]
        return delta_cont

    def fetch_window(self, ix):
        """

        inputs:
            ix: int, index of window function
        """
        # scale i'th window function
        if len(self.pred.shape) > 1:
            wnn_ix = 1.*self.pred[:, ix]
        else:
            wnn_ix = 1.*self.pred
        wnn_ix = wnn_ix / wnn_ix.mean()
        wnn_ix = wnn_ix.clip(0.5, 2.0)

        window = np.zeros(self.npix)
        window[self.hpix] = wnn_ix
        return window

#----
#Functions written by Tanveer
import pyccl as ccl
import pandas as pd

def cgll(ell, bias, **cosmo_kwargs):
    """Given a cosmology in pyccl generate clgg

    Inputs:
        b : linear bias
    """

    #define cosmology
    cosmo = ccl.Cosmology(**cosmo_kwargs)

    #read in dNdz
    dNdzddeg2 = pd.read_csv("../dat/nz_blanc.txt", sep=",")
    zmid = dNdzddeg2['Redshift_mid']
    dndz = dNdzddeg2['dndz/deg^2'] * 14000
    dn = dndz[:-1] * np.diff(zmid)  #redshift bin width

    #set constant bias
    b = bias*np.ones(len(zmid[:-1]))

    #Create CCL tracer object for galaxy clustering
    elg_ccl = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zmid[:-1], dn),
        bias=(zmid[:-1],b))

    #calculate theoretical Cls
    cls_elg_th = ccl.angular_cl(cosmo, elg_ccl, elg_ccl, ell)

    return cls_elg_th

import itertools
def bin_mat(r=[],mat=[],r_bins=[]):
    """Sukhdeep's Code to bins data and covariance arrays

    Input:
    -----
        r  : array which will be used to bin data, e.g. ell values
        mat : array or matrix which will be binned, e.g. Cl values
        bins : array that defines the left edge of the bins,
               bins is the same unit as r

    Output:
    ------
        bin_center : array of mid-point of the bins, e.g. ELL values
        mat_int : binned array or matrix
    """

    bin_center=0.5*(r_bins[1:]+r_bins[:-1])
    n_bins=len(bin_center)
    ndim=len(mat.shape)
    mat_int=np.zeros([n_bins]*ndim,dtype='float64')
    norm_int=np.zeros([n_bins]*ndim,dtype='float64')
    bin_idx=np.digitize(r,r_bins)-1
    r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
    dr=np.gradient(r2)
    r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
    dr=dr[r2_idx]
    r_dr=r*dr

    ls=['i','j','k','l']
    s1=ls[0]
    s2=ls[0]
    r_dr_m=r_dr
    for i in np.arange(ndim-1):
        s1=s2+','+ls[i+1]
        s2+=ls[i+1]
        r_dr_m=np.einsum(s1+'->'+s2,r_dr_m,r_dr)#works ok for 2-d case

    mat_r_dr=mat*r_dr_m
    for indxs in itertools.product(np.arange(min(bin_idx),n_bins),repeat=ndim):
        x={}#np.zeros_like(mat_r_dr,dtype='bool')
        norm_ijk=1
        mat_t=[]
        for nd in np.arange(ndim):
            slc = [slice(None)] * (ndim)
            #x[nd]=bin_idx==indxs[nd]
            slc[nd]=bin_idx==indxs[nd]
            if nd==0:
                mat_t=mat_r_dr[slc]
            else:
                mat_t=mat_t[slc]
            norm_ijk*=np.sum(r_dr[slc[nd]])
        if norm_ijk==0:
            continue
        mat_int[indxs]=np.sum(mat_t)/norm_ijk
        norm_int[indxs]=norm_ijk
    return bin_center,mat_int

#I/O of systematic maps
def read_img_map(filename, nside=1024):
    d = ft.read(filename)
    m = np.zeros(12*nside*nside)
    v = d['weight'] / np.median(d['weight']) # normalize by the median
    v = v.clip(0.5, 2.0)                     # clip the extremes
    v = v / v.mean()                         # normalize to one
    m[d['hpix']] = v
    return m

def contaminate_map(F, delta, mask, noisemap = None, additive = None,
    img_applied_data = False):
    """
    inputs:
        F (float array) : selection function in healpy
        delta (float array) :truth density map in healpy
        mask (bool array) : mask for density map in healpy
        noisemap (float array) : noise model for density map in healy
        additive (boolean) : flag for additive window component
        img_applied_data (boolean) : flag for applying systematics to data map
                                    than random maps
    """

    delta_cont = np.zeros(delta.shape[0])
    delta_cont[:] = hp.UNSEEN #default set to UNSEEN

    if noisemap is not None: #if noise map provided
        if additive is not None: #if additive component provided
            if(img_applied_data): #if systematics applied to data
                delta_cont[mask] = (F[mask]/additive[mask])*(1 + delta[mask]) + \
                np.sqrt(F[mask]/additive[mask])*noisemap[mask] - 1. #EXP F
            else:
                delta_cont[mask] = F[mask]*(1 + delta[mask]) + \
                np.sqrt(F[mask])*noisemap[mask] - additive[mask] #EXP C
        else:
            if(img_applied_data):
                delta_cont[mask] = delta[mask] + \
                np.sqrt(1/F[mask])*noisemap[mask] #EXP D and E
            else:
                delta_cont[mask] = F[mask]*delta[mask] + \
                np.sqrt(F[mask])*noisemap[mask] #EXP A and B
    else:
        delta_cont[mask] = F[mask]*delta[mask] #only true when img_applied_data
    return delta_cont

def cls_from_mock(cls_th, cls_shot_noise, F, mask, seed, LMAX, NSIDE = 1024, \
    additive = None, img_applied_data = False):
    """Generate a mock given conditions and calculate pseudo-Cls from the mock.

    Inputs:
        cls_th (np.array) : array of theory Cl values to be used to generate
                            mock
        cls_shot_noise (np.array) : array of Cl values to be used to generate
                                    noise mock
        F (np.array) : Imaging contaminant map from GenSys. Should be same size
                        as NSIDE
        mask (np.array) : Mask map. Should be same size as NSIDE
        seed (int) : seed for mock generation
        LMAX (int) : lmax to be calculated for pseudo Cl
        NSIDE (int) : nside for healpy
        additive (np.array) : array of average F map used for additive component
                                experiments
        img_applied_data (bool) : flag for applying systematics to data map
                                    than random maps


    Returns:
        cls_obs (np.array) : array of pseudo-Cls based on generated mock. Not
                            corrected for fsky
    """

    #generate overdensity signal mock
    np.random.seed(seed)
    print(seed)
    delta_g = hp.synfast(cls_th,
        nside = NSIDE, lmax = LMAX, pol=False, verbose=False)

    #generate noise mock
    np.random.seed(2*seed + 4029) #random different seed for noise
    print(2*seed + 4029)
    noise_g = hp.synfast(cls_shot_noise,
        nside = NSIDE, lmax = LMAX, pol = False, verbose = False)

    #add img sys
    if additive is not None:
        delta_c = contaminate_map(F = F, delta = delta_g, mask = mask,
        noisemap = noise_g, additive = additive,
        img_applied_data = img_applied_data)
    else:
        delta_c = contaminate_map(F = F, delta = delta_g, mask = mask,
        noisemap = noise_g, img_applied_data = img_applied_data)

    #calcuate pseudo-Cl
    cls_obs = hp.anafast(delta_c, lmax = LMAX -1, pol = False)

    return cls_obs
