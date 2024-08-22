NVCC = nvcc
CFLAGS = -std=c++11
INCDIR = 
LIBDIR  = 
LIB = -lm

bin/%: %.cu
	${NVCC} ${CFLAGS} ${INCDIR} ${LIBDIR} $< -o $@ ${LIB} 


all: bin/gemm_v01

.PHONY:all