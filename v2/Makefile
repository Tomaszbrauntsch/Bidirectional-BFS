# Makefile for MPI bidirectional-bitset BFS
# Assumes source file is mpi_bibfs_bitset.cpp

# MPI C++ compiler
MPICXX ?= mpicxx

# Compiler flags
CXXFLAGS  := -std=c++11 -O3
LDFLAGS   :=

# Executable name
TARGET    := second_try

# Source and object files
SRCS      := second_try.cpp
OBJS      := $(SRCS:.cpp=.o)

.PHONY: all run clean

all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(MPICXX) $(LDFLAGS) -o $@ $^

# Compile
%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

# Run with 4 ranks (adjust -n, bin-file, src, dst as needed)
run: all
	mpirun -np 4 ./$(TARGET) ../graphs/1k.bin 0 999

# Remove objects and executable
clean:
	rm -f $(OBJS) $(TARGET)
