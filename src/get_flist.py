"""Get list of window files and save them to pass to pCl_parallel.py"""

from glob import glob 
import numpy as np 

window_dir = "/mnt/gosling1/tkarim/img-sys/mocks/linear/windows/"
out_dir = "../dat/"

flist = glob(window_dir + "window_*")

np.save(out_dir + "flist_window_linear.npy", flist)
