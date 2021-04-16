#!/bin/bash --login
 
#SBATCH --job-name=test_matmul
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G
#SBATCH --output=%x-%j.SLURMout
  
cd ${PBS_O_WORKDIR}
  
module purge
module load cuDNN/7.5.0.56-CUDA-10.0.130
module load CUDA/10.0.130
module load GNU/5.4.0-2.26  OpenMPI/1.10.3

module load Python/3.5.2
source ~/tfenv/bin/activate
cd ~/DanQ/DanQ

python DanQ_train_1.py
