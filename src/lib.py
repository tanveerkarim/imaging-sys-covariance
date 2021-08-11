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

def contaminate_map(expname, F, delta, mask, noisemap = None, additive = None):
    """
    inputs:
        expname (str) : name of experiment 
        F (float array) : selection function in healpy
        delta (float array) :truth density map in healpy
        mask (bool array) : mask for density map in healpy
        noisemap (float array) : noise model for density map in healy
        additive (healpy map, optional) : additive window component
    """

    delta_cont = np.zeros(delta.shape[0])
    delta_cont[:] = hp.UNSEEN #default set to UNSEEN
    #print(expname)
    if((expname == 'A') | (expname == 'B')):
        delta_cont[mask] = F[mask]*delta[mask] + np.sqrt(F[mask])*noisemap[mask]
    elif(expname == 'C'):
        delta_cont[mask] = F[mask]*(1 + delta[mask]) + np.sqrt(F[mask])*noisemap[mask] - additive[mask]
    elif(expname == 'D'):
        delta_cont[mask] = delta[mask] + np.sqrt(1/F[mask])*noisemap[mask]
    elif(expname == 'E'):
        delta_cont[mask] = F[mask]/additive[mask]*delta[mask] + (np.sqrt(F[mask])/additive[mask])*noisemap[mask]
    elif(expname == 'F'):
        delta_cont[mask] = (F[mask]/additive[mask])*(1. + delta[mask]) + (np.sqrt(F[mask])/additive[mask])*noisemap[mask] - 1
    else:
        raise ValueError("Wrong experiment name entered.")
        
    #if noisemap is not None: #if noise map provided
    #    if additive is not None: #if additive component provided
    #        if(divide_window): #if systematics applied to data
    #            delta_cont[mask] = (F[mask]/additive[mask])*(1 + delta[mask]) + \
    #            (np.sqrt(F[mask])/additive[mask])*noisemap[mask] - 1. #EXP F
    #        else:
    #            delta_cont[mask] = F[mask]*(1 + delta[mask]) + \
    #            np.sqrt(F[mask])*noisemap[mask] - additive[mask] #EXP C
    #    else:
    #        if(divide_window):
    #            delta_cont[mask] = delta[mask] + \
    #            np.sqrt(1/F[mask])*noisemap[mask] #EXP D and E
    #        else:
    #            delta_cont[mask] = F[mask]*delta[mask] + \
    #            np.sqrt(F[mask])*noisemap[mask] #EXP A and B
    #else:
    #    delta_cont[mask] = F[mask]*delta[mask] #only true when img_applied_data
    return delta_cont

def cls_from_mock(expname, cls_th, cls_shot_noise, F, mask, seed, NSIDE = 1024, \
    additive = None):
    """Generate a mock given conditions and calculate pseudo-Cls from the mock.

    Inputs:
        expname (str) : name of experiment
        cls_th (np.array) : array of theory Cl values to be used to generate
                            mock
        cls_shot_noise (np.array) : array of Cl values to be used to generate
                                    noise mock
        F (np.array) : Imaging contaminant map from GenSys. Should be same size
                        as NSIDE
        mask (np.array) : Mask map. Should be same size as NSIDE
        seed (int) : seed for mock generation
        NSIDE (int) : nside for healpy
        additive (np.array) : array of average F map used for additive component
                                experiments

    Returns:
        cls_obs (np.array) : array of pseudo-Cls based on generated mock. Not
                            corrected for fsky
    """

    #generate overdensity signal mock
    #print(seed)
    np.random.seed(seed)
    delta_g = hp.synfast(cls_th,
        nside = NSIDE, lmax = 3*NSIDE-1, pol=False, verbose=False)

    #generate noise mock
    np.random.seed(2*seed) #random different seed for noise
    noise_g = hp.synfast(cls_shot_noise,
        nside = NSIDE, lmax = 3*NSIDE-1, pol = False, verbose = False)

    #add img sys
    delta_c = contaminate_map(expname = expname, F = F, delta = delta_g, 
                              mask = mask, noisemap = noise_g, 
                              additive = additive)
    #if additive is not None:
    #    delta_c = contaminate_map(F = F, delta = delta_g, mask = mask,
    #    noisemap = noise_g, additive = additive,
    #    divide_window = divide_window)
    #else:
    #    delta_c = contaminate_map(F = F, delta = delta_g, mask = mask,
    #    noisemap = noise_g, divide_window = divide_window)

    #calcuate pseudo-Cl
    cls_obs = hp.anafast(delta_c, lmax = LMAX -1, pol = False)

    return cls_obs

#cumulative SN calculation
def matrix_cut(mat=[],x=[]):
    """
    mat : covariance matrix
    x : l cuts to be applied to mat
    """
    m=mat[x]
    N=sum(x)
    m2=np.zeros((N,N))
    j=0
    for i in m:
        m2[j]=i[x]
        j=j+1
    return m2

def SN_cum(cov=[],lb=[],cl=[],diag=False,lmin=2,lmax=1e4,use_hartlap=False,nsim=1000):
    """
    cov : covariance matrix
    lb : bin centres
    cl : binned cl array
    """

    sni=np.zeros_like(lb)
    for i in np.arange(len(lb)):
        if lb[i]<lmin or lb[i]>lmax:
            continue
        x=lb<=lb[i]
        x*=lb>lmin
        cov2_cut=matrix_cut(mat=cov,x=x)
        if diag:
            cov2_cut=np.diag(np.diag(cov2_cut))
        cov2_cut_inv=np.linalg.inv(cov2_cut)

        cl_i=cl[x]
        SN2=cl_i@cov2_cut_inv@cl_i
        if use_hartlap:
            SN2*=(nsim-2-x.sum())/(nsim-1)
        sni[i]=SN2
    return np.sqrt(sni)

#Theory pCl functions
def set_window_here(ztomo_bins_dict={}, nside=1024, mask = None, unit_win=False, cmb=False):
    """
        This function sets the window functions for the datasets. 
        These windows are necessary for converting cl to pseudo-cl.
    """
    #FIXME: make sure nside, etc. are properly matched. if possible, use same nside for cmb and galaxy maps. Use ud_grade where necessary.
    for i in np.arange(ztomo_bins_dict['n_bins']):
        if unit_win:
            cl_map=hp.ma(np.ones(12*nside*nside))
            cl_i=1
        elif cmb:
            #cl_map=np.load('/mnt/store1/tkarim/mask_2048.fits') #FIXME: add the CMB lensing window here.
            #window_map=np.load("/mnt/store1/boryanah/AbacusSummit_base_c000_ph006/light_cones/mask_edges_ring_2048.npy")
            #window_map=np.load("/home/tkarim/imaging-sys-covariance/dat/windows/Favg/Favg_map_unpickled.npy")
            #window_map = window_map.astype(np.float64)
            #window_map_noise = window_map
            #mask = cl_map
            print("cmb")
        else:
            window_map=np.load("/home/tkarim/imaging-sys-covariance/dat/windows/Favg/Favg_map_unpickled.npy") #randoms are the window function.
            #window_map=np.load("/mnt/store1/boryanah/AbacusSummit_base_c000_ph006/light_cones/mask_edges_ring_2048.npy") #randoms are the window function.
            window_map = window_map.astype(np.float64)
            window_map_noise = np.sqrt(window_map)
        
        if mask is None:
            mask=window_map>0 #FIXME: input proper mask if possible
        window_map[window_map<0]=0 #numerical issues can make this happen
        window_map/=window_map[mask].mean() #normalize window to 1
        window_map[~mask]=hp.UNSEEN
        window_map_noise[~mask]=hp.UNSEEN
        
        ztomo_bins_dict[i]['window']=window_map
        ztomo_bins_dict[i]['window_N']=window_map_noise #window of noise 

    return ztomo_bins_dict

def zbin_pz_norm(ztomo_bins_dict={},tomo_bin_indx=None,zbin_centre=None,p_zspec=None,ns=0,bg1=1,
                 mag_fact=0,k_max=0.3):
    """
        This function does few pre-calculations and sets some parameters for datasets that 
        will be input into skylens.
    """

    dzspec=np.gradient(zbin_centre) if len(zbin_centre)>1 else 1 #spec bin width

    if np.sum(p_zspec*dzspec)!=0:
        p_zspec=p_zspec/np.sum(p_zspec*dzspec) #normalize histogram
    else:
        p_zspec*=0
    nz=dzspec*p_zspec*ns

    i=tomo_bin_indx
    x= p_zspec>-1 #1.e-10; incase we have absurd p(z) values

    ztomo_bins_dict[i]['z']=zbin_centre[x]
    ztomo_bins_dict[i]['dz']=np.gradient(zbin_centre[x]) if len(zbin_centre[x])>1 else 1
    ztomo_bins_dict[i]['nz']=nz[x]
    ztomo_bins_dict[i]['ns']=ns
    ztomo_bins_dict[i]['W']=1. #redshift dependent weight
    ztomo_bins_dict[i]['pz']=p_zspec[x]*ztomo_bins_dict[i]['W']
    ztomo_bins_dict[i]['pzdz']=ztomo_bins_dict[i]['pz']*ztomo_bins_dict[i]['dz']
    ztomo_bins_dict[i]['Norm']=np.sum(ztomo_bins_dict[i]['pzdz'])
    ztomo_bins_dict[i]['b1']=bg1 # FIXME: this is the linear galaxy bias. Input proper values. We can also talk about adding other bias models if needed.
    ztomo_bins_dict[i]['bz1'] = None #array; set b1 to None if passing redz dependent bias 
    ztomo_bins_dict[i]['AI']=0. # this will be zero for our project
    ztomo_bins_dict[i]['AI_z']=0. # this will be zero for our project
    ztomo_bins_dict[i]['mag_fact']=mag_fact  #FIXME: You need to figure out the magnification bias prefactor. For example, see appendix B of https://arxiv.org/pdf/1803.08915.pdf
    ztomo_bins_dict[i]['shear_m_bias'] = 1.  #
    
    #convert k to ell
    zm=np.sum(ztomo_bins_dict[i]['z']*ztomo_bins_dict[i]['pzdz'])/ztomo_bins_dict[i]['Norm']
    ztomo_bins_dict[i]['lm']=k_max*cosmo_h.comoving_transverse_distance(zm).value 
    return ztomo_bins_dict

def source_tomo_bins(zphoto_bin_centre=None,p_zphoto=None,ntomo_bins=None,ns=26,
                     zspec_bin_centre=None,n_zspec=100,ztomo_bins=None,
                     f_sky=0.3,nside=256,use_window=False,
                    bg1=1,l=None,mag_fact=0,
                    k_max=0.3,use_shot_noise=True,**kwargs):
    """
        Setting galaxy redshift bins in the format used in skylens code.
        Need
        zbin_centre (array): redshift bins for every source bin. if z_bins is none, then dictionary with
                    with values for each bin
        p_zs: redshift distribution. same format as zbin_centre
        z_bins: if zbin_centre and p_zs are for whole survey, then bins to divide the sample. If
                tomography is based on lens redshift, then this arrays contains those redshifts.
        n_gal: number density for shot noise calculation
        n_zspec : number of histogram bins in spectroscopic dndz (if zspec_bin_centre is not passed)
        ztomo_bins : edges of tomographic bins in photometric redshift (assign galaxies to tomo bins using photz)
                    e.g. [0.6, 1., 1.6]
        k_max : cut in k-space; CHECK FOR BUG
    """
    ztomo_bins_dict={} #dictionary of tomographic bins

    if ntomo_bins is None:
        ntomo_bins=1

    if ztomo_bins is None:
        ztomo_bins=np.linspace(min(zphoto_bin_centre)-0.0001,max(zphoto_bin_centre)+0.0001,ntomo_bins+1)
    if zspec_bin_centre is None: #histogram of dndz; defines bin centres
        zspec_bin_centre=np.linspace(0,max(ztomo_bins)+1,n_zspec)
    dzspec=np.gradient(zspec_bin_centre)
    dzphoto=np.gradient(zphoto_bin_centre) if len(zphoto_bin_centre)>1 else [1]
    zphoto_bin_centre=np.array(zphoto_bin_centre)

    #zl_kernel=np.linspace(0,max(zbin_centre),50) #galaxy position kernel; identical to b*dndz
    #lu=Tracer_utils() 
    #cosmo_h=cosmo_h_PL #cosmology parameters in astropy convention; default is Skylens default

    zmax=max(ztomo_bins)

    l=[1] if l is None else l
    ztomo_bins_dict['SN']={} #shot noise dict
    ztomo_bins_dict['SN']['galaxy']=np.zeros((len(l),ntomo_bins,ntomo_bins)) # ell X no. of tomo bins X no. of tomo bins 
    ztomo_bins_dict['SN']['kappa']=np.zeros((len(l),ntomo_bins,ntomo_bins))

    for i in np.arange(ntomo_bins):
        ztomo_bins_dict[i]={}
        indx=zphoto_bin_centre.searchsorted(ztomo_bins[i:i+2]) #find bins that belong to this photometric bin

        if indx[0]==indx[1]: #if only one bin
            indx[1]=-1
        zbin_centre=zphoto_bin_centre[indx[0]:indx[1]]
        p_zspec=p_zphoto[indx[0]:indx[1]] #assuming spectroscopic and photometric dndz are same; CHANGE IF NOT 
        nz=ns*p_zspec*dzphoto[indx[0]:indx[1]]
        ns_i=nz.sum()

        ztomo_bins_dict = zbin_pz_norm(ztomo_bins_dict=ztomo_bins_dict, tomo_bin_indx=i, 
                                       zbin_centre=zbin_centre,
                                       p_zspec=p_zspec,ns=ns_i,bg1=bg1, mag_fact=mag_fact,k_max=k_max)
        
        zmax=max([zmax,max(ztomo_bins_dict[i]['z'])])
        if use_shot_noise:
            ztomo_bins_dict['SN']['galaxy'][:,i,i]=galaxy_shot_noise_calc(zg1=ztomo_bins_dict[i],
                                                                  zg2=ztomo_bins_dict[i])
            #the following is set in the CMB lensing bin
            #zs_bins['SN']['kappa'][:,i,i]=shear_shape_noise_calc(zs1=zs_bins[i],zs2=zs_bins[i],
            #                                                     sigma_gamma=sigma_gamma) #FIXME: This is almost certainly not correct

    ztomo_bins_dict['n_bins']=ntomo_bins #easy to remember the counts
    #ztomo_bins_dict['z_lens_kernel']=zl_kernel
    ztomo_bins_dict['zmax']=zmax
    ztomo_bins_dict['zp']=zphoto_bin_centre
    ztomo_bins_dict['pz']=p_zphoto
    ztomo_bins_dict['z_bins']=ztomo_bins
    
    if use_window:
        ztomo_bins_dict=set_window_here(ztomo_bins_dict=ztomo_bins_dict,nside=nside, unit_win=False)
    return ztomo_bins_dict

def DESI_elg_bins(ntomo_bins=1, f_sky=0.3,nside=256,use_window=True, bg1=1, 
                       l=None, mag_fact=0,ztomo_bins=None,**kwargs):

    home='/home/tkarim/'
    fname='nz_blanc+abacus.txt'
#     t=np.genfromtxt(home+fname,names=True,skip_header=3)
    #t=np.genfromtxt(home+fname,names=True)
    t = pd.read_csv(home + fname)
    dz=t['Redshift_mid'][2]-t['Redshift_mid'][1]
    zmax=max(t['Redshift_mid'])+dz/2
    zmin=min(t['Redshift_mid'])-dz/2

    z=t['Redshift_mid']
    
    pz=t['dndz/deg^2']
    
    ns=np.sum(pz)
    d2r = 180/np.pi
    ns/=d2r**2 #convert from deg**2 to rd**2

    if ztomo_bins is None: #this defines the bin edges if splitting the sample into bins. Preferably pass it as an argument whenusing multiple bins.
        ztomo_bins=np.linspace(zmin, min(2,zmax), ntomo_bins+1) #define based on experiment
    print(zmin,zmax,ztomo_bins,ns)
    return source_tomo_bins(zphoto_bin_centre=z, p_zphoto=pz, ns=ns, ntomo_bins = ntomo_bins,
                            mag_fact=mag_fact, ztomo_bins=ztomo_bins,f_sky=f_sky,nside=nside,
                            use_window=use_window,bg1=bg1, l=l,**kwargs)