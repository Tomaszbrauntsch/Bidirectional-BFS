// mpi_bas.cpp
#include <mpi.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

extern "C" void gpuSetup(
    int, int,
    int*, int*,
    int, int,
    int, int,
    int*, int*, int*, int*, int*, int*
);
extern "C" void gpuIterate(unsigned char*, unsigned char*, unsigned char*);
extern "C" void gpuCopyHostToDevice();
extern "C" void gpuFinalize();

int main(int argc, char** argv) {
  MPI_Init(&argc,&argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // DEBUG: confirm rank/size before GPU
  MPI_Barrier(MPI_COMM_WORLD);
  printf("[DEBUG BEFORE GPU] rank=%d size=%d\n", rank, size);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  if (argc != 4) {
    if (rank == 0) std::cerr<<"Usage: mpirun -n <p> ./mpi_bibfs <graph.bin> <src> <dst>\n";
    MPI_Finalize();
    return 1;
  }

  const char* filename = argv[1];
  int SRC = std::atoi(argv[2]);
  int DST = std::atoi(argv[3]);

  uint32_t N, M;
  std::vector<uint32_t> flat;
  if (rank == 0) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(&N), 4);
    in.read(reinterpret_cast<char*>(&M), 4);
    flat.resize(2*M);
    in.read(reinterpret_cast<char*>(flat.data()), 2*M*4);
  }
  MPI_Bcast(&N, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&M, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  if (rank != 0) flat.resize(2*M);
  MPI_Bcast(flat.data(), 2*M, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // build CSR
  std::vector<int> deg(N, 0);
  for (int i = 0; i < 2*(int)M; i += 2) deg[flat[i]]++;
  std::vector<int> row_ptr(N+1, 0);
  for (int i = 0; i < (int)N; i++) row_ptr[i+1] = row_ptr[i] + deg[i];
  std::vector<int> col_ind(2*M), cur = row_ptr;
  for (int i = 0; i < 2*(int)M; i += 2) {
    int u = flat[i], v = flat[i+1];
    col_ind[cur[u]++] = v;
  }

  // host BFS arrays
  std::vector<int> h_vis_s(N,0), h_vis_t(N,0);
  std::vector<int> h_front_s(N,0), h_front_t(N,0);
  std::vector<int> h_pred_s(N,-1), h_pred_t(N,-1);
  h_vis_s[SRC]=1; h_front_s[SRC]=1; h_pred_s[SRC]=SRC;
  h_vis_t[DST]=1; h_front_t[DST]=1; h_pred_t[DST]=DST;

  int threads = 256;
  int blocks  = (N + threads - 1)/threads;

  gpuSetup(
    N, 2*M,
    row_ptr.data(), col_ind.data(),
    SRC, DST,
    threads, blocks,
    h_vis_s.data(), h_vis_t.data(),
    h_front_s.data(), h_front_t.data(),
    h_pred_s.data(), h_pred_t.data()
  );

  // DEBUG: confirm rank/size after gpuSetup
  MPI_Barrier(MPI_COMM_WORLD);
  printf("[DEBUG AFTER gpuSetup] rank=%d size=%d\n", rank, size);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  bool global_hit = false;
  while (!global_hit) {
    unsigned char schg, tchg, inter;

    // DEBUG: before iterate
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[DEBUG BEFORE gpuIterate] rank=%d\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    gpuIterate(&schg, &tchg, &inter);

    // DEBUG: after iterate
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[DEBUG AFTER gpuIterate] rank=%d\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // merge on host
    MPI_Allreduce(MPI_IN_PLACE, h_vis_s.data(),   N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_vis_t.data(),   N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_front_s.data(), N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_front_t.data(), N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_pred_s.data(),  N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_pred_t.data(),  N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // push updates back
    gpuCopyHostToDevice();

    int local_hit = inter ? 1 : 0;
    MPI_Allreduce(&local_hit, &global_hit, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (!schg && !tchg) break;
  }

  if (rank == 0) {
    std::cout << "Running with " << size << " ranks\n";
    int meet = -1;
    for (int i = 0; i < (int)h_vis_s.size(); i++){
      if (h_vis_s[i] && h_vis_t[i]){ meet = i; break; }
    }
    if (meet < 0) {
      std::cout << "No path found\n";
    } else {
      std::vector<int> path;
      for (int x = meet; x != SRC; x = h_pred_s[x]) path.push_back(x);
      path.push_back(SRC);
      std::reverse(path.begin(), path.end());
      int cur = meet;
      while (cur != DST) {
        cur = h_pred_t[cur];
        path.push_back(cur);
      }
      std::cout << "Path:";
      for (int v: path) std::cout << " " << v;
      std::cout << "\n";
    }
  }

  gpuFinalize();
  MPI_Finalize();
  return 0;
}
