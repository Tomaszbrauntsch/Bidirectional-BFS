# Bidirectional-BFS
Made by Hamdi Korreshi and Tomasz Brauntsch
Utilizing CUDA and MPI to perform bidirectional BFS on a cluster of two machines, to find the shortest path between two the source and destination nodes. We use 0 (src) and (dst) n-1 to make the scripting easier, but I have confirmed that other node options work. Make sure to download the requirements.txt to download the libraries needed.
# V1 - Serial
The serial version is a baseline for this project to see how much faster the Bi-BFS went. (Spoiler Serial is the best currently)

# V2 - MPI only
The MPI version only supports CPU processing which shows disappointing results.

# V3 - Cuda only
The CUDA only version can somewhat compete with the serial version.

# V4 - MPI & Cuda
The MPI + CUDA should be the fastest but there are many asterisks on that claim. The limitations sections cover this.
Use this to test, if you don't want to use the script: mpirun -np 4 -hostfile host_file mpi_bibfs <1000k.bin> 0 <end>.

# Limitations
The TLDR; reason for the parallelized versions being so much slower all comes down to 2 main reasons, graphs are too small and the hardware I used these tests on. A graph of of 10 million node would be better made to show the difference between them, also a path that is in the 4 digits should show completely different results. Thus if you are on the next semester or someone else that wants to try this, DO NOT USE NETWORKX, switch to igraph or gml. They are much better and do not require 900 GBs to generate a a 1 million node graph. We didn't have the best hardware, we were provided 2 laptops with Quadro M1200, but the real problem was the switch. The switch we have is only a 1GB switch which is not able to even handle a 100k node graph properly, so if you want to try something similar get a good switch since the overhead of communication and sending data back and forth is a giant amount.

# File Structure
.
├── .gitignore
├── README.md
├── benchmark_results.csv
├── benchmark_table.txt
├── benchmark_test.sh
├── env
│   ├── bin
│   ├── include
│   ├── lib
│   ├── lib64 -> lib
│   ├── pyvenv.cfg
│   └── share
├── graphs
│   ├── 100k.json
│   ├── 10k.json
│   ├── 1k.json
│   ├── 50k.json
│   ├── benchmark_results.csv
│   ├── generate_graph.py
│   ├── make_graphs
│   └── read_graph.py
├── requirements.txt
├── single_machine_bench.sh
├── v1
│   ├── Makefile
│   ├── bibfs_serial
│   ├── main-v1.cpp
│   └── mpi-skeleton
├── v2
│   ├── 10k.json
│   ├── 1k.json
│   ├── 50k.json
│   ├── Makefile
│   ├── generate_graph.py
│   ├── main-v2.cpp
│   ├── mpi-draft
│   ├── read_in.cpp
│   ├── second_try
│   ├── second_try.cpp
│   └── show_matrix
├── v3
│   ├── Makefile
│   ├── bibfs_cuda_only
│   ├── bibfs_cuda_only.cu
│   └── cuda-skeleton-with-mpi
├── v4
│   ├── Makefile
│   ├── comp.cu
│   ├── comp.h
│   ├── main-v4.cpp
│   ├── mpi_bas.cpp
│   └── mpi_bibfs
└── visual
    ├── m3.py
    ├── main.sh
    └── test.png

12 directories, 44 files
