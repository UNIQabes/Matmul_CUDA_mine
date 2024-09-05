NVCC = nvcc
CXXFLAGS = -std=c++11 
CUFLAGS = -arch=sm_70
INCDIR = 
LIBDIR = 
LIB = 

bin/%: %.cu
	${NVCC} ${CXXFLAGS} ${CUFLAGS} ${INCDIR} ${LIBDIR} $< -o $@ ${LIB} 


all: bin/main_fp32

.PHONY:all