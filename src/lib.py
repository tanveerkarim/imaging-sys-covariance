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

    def contaminate(self, window, delta, mask, noisemap = None, additive = None):
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
                delta_cont[mask_] = (1 + delta[mask_])*window[mask_] + \
                    noisemap[mask_]*np.sqrt(window[mask_]) - additive[mask_]
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
