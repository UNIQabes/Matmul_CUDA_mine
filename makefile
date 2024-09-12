NVCC = nvcc
CXXFLAGS = -std=c++11 
CUFLAGS = -arch=sm_80
INCDIR = 
LIBDIR = 
LIB = 



bin/%: %.cu
	${NVCC} ${CXXFLAGS} ${CUFLAGS} ${INCDIR} ${LIBDIR} $< -o $@ ${LIB} 

bin/main_fp32_sm70: main_fp32.cu
	${NVCC} ${CXXFLAGS} -arch=sm_70 ${INCDIR} ${LIBDIR} $< -o $@ ${LIB} 


all: bin/main_fp32 bin/main_fp32_sm70

.PHONY:all