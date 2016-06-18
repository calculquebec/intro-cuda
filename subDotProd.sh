#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=00:10:00
#PBS -N dotprod

module load CUDA/7.5.18
 
cd $PBS_O_WORKDIR
./dotprod
