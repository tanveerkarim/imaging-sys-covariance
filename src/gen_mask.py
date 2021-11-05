import healpy as hp
import numpy as np

fpix = hp.read_map("/home/tkarim/imaging-sys-covariance/dat/fpix_map.hp1024.fits")
fracPixThreshold = 0.5 #threshold above which pixels are good. good == 1
mask_bool = fpix > 0.5

np.save("/home/tkarim/imaging-sys-covariance/dat/mask_bool_dr9.npy",
        mask_bool, allow_pickle=False)
