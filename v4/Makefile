################################################################################
# Hybrid MPI+CUDA BI‑BFS Makefile                                              #
################################################################################

# CUDA toolkit location
CUDA_PATH ?= /usr/local/cuda

# Host compiler (for NVCC’s -ccbin)
HOST_COMPILER ?= g++

# NVCC invocation
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# MPI C++ compiler wrapper
MPICXX ?= mpicxx

# Compute capabilities you want to support
SMS ?= 50 52 60 61 70 75 80 86 89 90
GENCODE_FLAGS := \
  $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm)) \
  -gencode arch=compute_$(lastword $(sort $(SMS))),code=compute_$(lastword $(sort $(SMS)))

# NVCC flags
NVCCFLAGS := --std=c++11

# MPI flags (if any)
MPIFLAGS := -std=c++11

# Include paths (e.g. for comp.h)
INCLUDES := -I./

# CUDA runtime library
LIB_CUDA := -L$(CUDA_PATH)/lib64 -lcudart

# Sources
CU_SRC    := comp.cu
MPI_SRC   := mpi_bas.cpp

# Objects
CU_OBJ    := $(CU_SRC:.cu=.o)
MPI_OBJ   := $(MPI_SRC:.cpp=.o)

# Final target
TARGET    := mpi_bibfs

.PHONY: all clean run

all: $(TARGET)

# 1) Compile comp.cu → comp.o
$(CU_OBJ): $(CU_SRC)
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

# 2) Compile mpi_bas.cpp → mpi_bas.o
$(MPI_OBJ): $(MPI_SRC)
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c $< -o $@

# 3) Link into mpi_bibfs
$(TARGET): $(CU_OBJ) $(MPI_OBJ)
	$(MPICXX) $(MPIFLAGS) $(MPI_OBJ) $(CU_OBJ) -o $@ $(LIB_CUDA)

# 4) Convenience: run with MPI (adjust -n to your process count)
run: $(TARGET)
	mpirun -np 4 ./$(TARGET) 1k.bin 0 999

clean:
	rm -f $(CU_OBJ) $(MPI_OBJ) $(TARGET)
