#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=00:10:00
#PBS -V
#PBS -N addNumbers

module add CUDA_Toolkit/7.5
 
cd $PBS_O_WORKDIR
./addNumbers