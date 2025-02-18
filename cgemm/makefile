BINARY_NAME = zgemm
CUDA_PATH   = /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda
MATH_LIB = /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4
CC          = nvcc -arch=sm_80 #--ptxas-options=-v
 
CFLAGS      = -O3 -std=c++11 
LDFLAGS     = -L$(MATH_LIB)/lib64 -lcudart -lcublas -lcurand

INCFLAGS    = -I$(CUDA_PATH)/include -ICommon -I. 


SRC         = $(wildcard *.cu)
build : $(BINARY_NAME)

$(BINARY_NAME): %: main.cu  utils/utils.cu 
# $(BINARY_NAME): %: kernel/%/sgemm.cu  utils/utils.cu 
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS)  $^   -o $@ 

clean:
	rm $(BINARY_NAME)

