################################################################################
# CUDA-only BFS Makefile                                                      #
# Adapted from NVIDIA Samples to build bibfs_cuda_only.cu                      #
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# Host compiler (falls back to g++)
HOST_COMPILER ?= g++

# NVCC with host compiler binding
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# Architecture and code generation flags (compute capability)
SMS ?= 50 52 60 61 70 75 80 86 89 90
GENCODE_FLAGS := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm)) \
                 -gencode arch=compute_$(lastword $(sort $(SMS))),code=compute_$(lastword $(sort $(SMS)))

# Common NVCC flags (adjust -std as needed)
NVCCFLAGS := --std=c++11 --threads 0

# Include paths
INCLUDES := -I../../Common

# Libraries
LIBRARIES := -L$(CUDA_PATH)/lib -L$(CUDA_PATH)/lib64 -lcudart

# Source and target
CU_SRC := bibfs_cuda_only.cu
OBJ    := $(CU_SRC:.cu=.o)
TARGET := bibfs_cuda_only

# Default build target
all: $(TARGET)

# Compile CUDA source to object
$(OBJ): $(CU_SRC)
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

# Link object to executable
$(TARGET): $(OBJ)
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) $^ -o $@ $(LIBRARIES)

# Run the executable (optional)
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all run clean

