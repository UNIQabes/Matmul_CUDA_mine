#!/bin/zsh
#SBATCH -p a100@40x4 
#SBATCH -N 1
#SBATCH -o report/%J_out.txt
#SBATCH -e report/%J_err.txt

#bin/gemm_v01
#nsys profile -o profile/report1 bin/gemm_v02
#bin/main_fp32

for i in $(seq 5)
do
	bin/main_fp32
done