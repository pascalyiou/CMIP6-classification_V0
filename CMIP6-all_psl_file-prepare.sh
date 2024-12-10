#!/bin/env bash
## Extracting North Atlantic region in psl in CMIP6 simulations
## Assumes that this code is stored in:
## ${HOME}/programmes/RStat/CMIP6class/CMIP6-all_psl_file-prepare.sh
## Batch launch (on spiritx) with:
## sbatch --partition zen4 --ntasks=1 --cpus-per-task=10 --time 60:00:00 ${HOME}/programmes/RStat/CMIP6class/CMIP6-all_psl_file-prepare.sh

## Pascal Yiou, LSCE, October 2024
## This code is distributed "as is" under a CeCILL license:
## http://www.cecill.info/
## It can be downloaded and used for free for academic purposes.
## For commercial uses, please contact Pascal Yiou (pascal.yiou@lsce.ipsl.fr)

## Environment configuration
#PBS -N CMIP6_download
#PBS -l nodes=1:ppn=1
#PBS -q week
#PBS -j oe
#PBS -m abe

## Use sbatch (on spiritx @ IPSL)
# Partition
#SBATCH --partition zen4
# Only one cpu 
#SBATCH --ntasks 1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
# asking time could be in minute 120 or 2:00:00  or 1-00:00:00(24H)
#SBATCH --time 60:00:00 
# 
# to debug script 
## set -x
# purging all module to be sure to not having interferaence with current environnement
module purge
# loading only needed module for sample
module load cdo/2.0.6

## Your user name
owner=`whoami`

# slurm in environment variable SLURM_CPUS_PER_TASK
export NUM_CPUS=$SLURM_CPUS_PER_TASK

##r0=("CMIP" "ScenarioMIP" "ScenarioMIP" "ScenarioMIP" "ScenarioMIP")
##r1=("historical" "ssp126" "ssp245" "ssp370" "ssp585")
r0=("CMIP")
r1=("historical")

yr1=1970
yr2=2000
var=psl

## Starting job loops
## Path for CMIP6 data on spiritx @ IPSL
cd /bdd/CMIP6

## Search over historical and four SSP scenarios, in lists defined by r0 and r1
ii=1
for i in "${!r0[@]}"; do
    echo "Starting process of ${r1[i]} files"
    
    for institute in $(ls ${r0[$i]}); do
##	echo $institute
        for model in $(ls ${r0[$i]}/$institute); do
##	    echo $institute $model
#Check if model has desired scenario
            if [ ! -d ${r0[$i]}/$institute/$model/${r1[$i]} ]; then
                continue
            fi

            for run in $(ls ${r0[$i]}/$institute/$model/${r1[$i]}); do
##		echo $institute $model $run
               	cd /bdd/CMIP6 
                r=${run%i*}
                r=${r:1}

                path="${r0[i]}"/$institute/$model/"${r1[i]}"/$run/day/$var/*/latest
##		echo $path
#Check if path exists
                if [ ! -d $path ]; then
                    continue
                fi

                cd $path
## Distribute tasks among available CPUs
		task_id=$(( (ii - 1) % NUM_CPUS + 1 ))
		echo Processing: ${institute} ${model} ${run}
		${HOME}/programmes/RStat/CMIP6class/CMIP6_psl_file-prepare_v2.sh ${institute} ${model} ${run} $yr1 $yr2 
		# if (( task_id == NUM_CPUS )); then
		#     # # Wait for tasks to complete before starting new ones
		#     wait
		# fi
		(( ii++ ))
		cd /bdd/CMIP6
            done
	    cd /bdd/CMIP6
        done
    done    
done

wait
## End job
