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
