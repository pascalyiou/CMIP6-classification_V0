#!/bin/bash -l
## Lancement en BATCH de classifications de modeles par tensorflow
## Version V1
## Pascal Yiou (LSCE), Nov. 2024, Dec. 2024
## Se lance sur HAL par:
## sbatch ${HOME}/programmes/RStat/CMIP6class/V0/CMIP6_classif-v0.sh NMOD 

# Instructions SBATCH always at the beginning of the script!

# Change the working directory before the execution of the job.
# Warning: the environment variables, e.g. $HOME,
# are not interpreted for the SBATCH instructions.
# Writing absolute paths is recommended.
#SBATCH -D /home/pyiou/programmes/RStat/CMIP6class/V0/

# The job partition (maximum elapsed time of the job).
##SBATCH --partition=day  

# The name of the job.
#SBATCH -J CMIP6_classif-v1.sh

# The number of GPU cards requested.
#SBATCH --gres=gpu:1

# Email notifications (e.g. the beginning and the end of the job).
#SBATCH --mail-user=pascal.yiou@lsce.ipsl.fr
#SBATCH --mail-type=all

# The path of the job log files.
# The error and the output logs can be merged into the same file.
# %j implements a job counter.
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out

## git branch -M V1

# Overtakes the system memory limits.
ulimit -s unlimited

# slurm in environment variable SLURM_CPUS_PER_TASK
export NUM_CPUS=$SLURM_CPUS_PER_TASK

JOBID=${SLURM_JOB_ID}

start_date=`date +"%m/%d/%Y (%H:%M)"`
echo -e "\n\nStarting script at: ${start_date}\n"

module purge
module load R/4.4.1 tensorflow/2.15.0

## Nombre de modèles
NMOD=$1

## Nombre de classifications
NSIM=50

R CMD BATCH "--args JJA ${NMOD} ${NSIM} ${JOBID}" ${HOME}/programmes/RStat/CMIP6class/V0/CMIP6_tensorflow-classif_v0.R
R CMD BATCH "--args DJF ${NMOD} ${NSIM} ${JOBID}" ${HOME}/programmes/RStat/CMIP6class/V0/CMIP6_tensorflow-classif_v0.R
R CMD BATCH "--args MAM ${NMOD} ${NSIM} ${JOBID}" ${HOME}/programmes/RStat/CMIP6class/V0/CMIP6_tensorflow-classif_v0.R
R CMD BATCH "--args SON ${NMOD} ${NSIM} ${JOBID}" ${HOME}/programmes/RStat/CMIP6class/V0/CMIP6_tensorflow-classif_v0.R


end_date=`date +"%m/%d/%Y (%H:%M)"`
echo -e "\nScript finished at: ${end_date}\n"
