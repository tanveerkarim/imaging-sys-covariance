


import numpy as np
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

    def contaminate(self, ix, delta, mask):
        """
        
        inputs:
            ix: int, index of the window
            delta: float array, truth density contrast            
            mask: bool array, mask for density contrast            
        """
        window = self.fetch_window(ix)  

        mask_ = mask & self.mask                
        delta_cont = np.zeros(self.npix)    
        delta_cont[:] = np.nan
        delta_cont[mask_] = (delta[mask_]+1.0)*window[mask_] -1.0        
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