#!/bin/zsh
#SBATCH -p bdw2-mixed
#SBATCH -N 1
#SBATCH -o report/%J_out.txt
#SBATCH -e report/%J_err.txt

#bin/gemm_v01
#nsys profile -o profile/report1 bin/gemm_v02
bin/main_fp32