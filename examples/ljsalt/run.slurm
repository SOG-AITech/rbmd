#!/bin/bash
#SBATCH -J rbmd-cuda                      
#SBATCH -p gpu_4090                    
#SBATCH -N 1                          
#SBATCH --gpus=3
#SBATCH --output=4090.out
module purge && module load  nccl/2.26_cuda12.8_4090 cuda/12.8   openmpi/4.1.5_cuda12.8   ucx/1.12.1_cuda12.8 cmake/4.2.0  gcc/9.3.0 
srun /data/home/scxj268/rbmd-mpi/build/rbmd -j ./ljsalt.json

