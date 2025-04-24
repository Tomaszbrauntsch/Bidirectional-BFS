NVCC    := nvcc
CXXFLAGS:= -O3 -std=c++14
LDFLAGS := -lcudart

all: bibfs_cuda_only

bibfs_cuda_only: bibfs_cuda_only.cu
	$(NVCC) $(CXXFLAGS) bibfs_cuda_only.cu -o $@ $(LDFLAGS)

clean:
	rm -f bibfs_cuda_only
