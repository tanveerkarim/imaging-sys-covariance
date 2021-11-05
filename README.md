# Modeling Effect of Imaging Systematics on Cosmological Parameter Estimation

How code works:
1. run `get_flist.py` -- generates list of window function locations
input - window function directory, output - npy file with locaitons.
2. run `sysavg_map.py` -- generates average systematic map based on mocks.
input - `flist.npy` file, output - `sysavg_map.npy` file.
3. run `gen_mask.py` -- generates boolean mask defining survey geometry.
input - `fpix.fits` containing fractional pixel map, output - `mask.npy` 
4. run `pcl_parallel_slurm.sh` -- generates stats in parallel by using 
`pCl_parallel.py`. input -  `flist.npy`, `sysavg_map.npy`, `mask.npy`
output - pcl, fsky and n_window.
