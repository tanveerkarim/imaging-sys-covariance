"""Get list of window files and save them to pass to pCl_parallel.py"""

from glob import glob 
import numpy as np 
import argparse

#parser for bash arguments
parser = argparse.ArgumentParser()
parser.add_argument("--window_type", "-wt", type=str, help="linear or nn")
args = parser.parse_args()
wtype = args.window_type

window_dir = "/mnt/gosling1/tkarim/img-sys/mocks/" + wtype + "/"
out_dir = "../dat/"

flist = glob(window_dir + "*fits")

np.save(out_dir + "flist_window_" + wtype + ".npy", flist)