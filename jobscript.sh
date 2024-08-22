#!/bin/zsh
#SBATCH -p bdw2-mixed
#SBATCH -N 3
#SBATCH -o report/%J_out
#SBATCH -e report/%J_err

bin/gemm_v01