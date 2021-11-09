#!/bin/bash -l
#SBATCH -q RM
#SBATCH -J mc
#SBATCH -e /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/mcmc_many%A_%a.err ###write out errors
#SBATCH -o /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/mcmc_many%A_%a.out ###write out print logs
#SBATCH -t 20:00:00
#SBATCH -N 10
#SBATCH --ntasks-per-node=27 ##look up haswell
#SBATCH --mem=128G ##request full node memory
#SBATCH -A phy200040p ##desi account charge
#SBATCH --array=1-10

ID=$SLURM_ARRAY_JOB_ID


script_home='/verafs/scratch/phy200040p/sukhdeep/project/skylens/scripts/'
outp_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/temp/'

pythonparser 

python $script_home/pcl_parallel.py --jobId $ID --output $outp_home