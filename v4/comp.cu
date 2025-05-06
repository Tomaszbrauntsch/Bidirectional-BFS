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
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 4) {
    if (rank == 0)
      std::cerr << "Usage: mpirun -n <p> ./mpi_bibfs <graph.bin> <src> <dst>\n";
    MPI_Finalize();
    return 1;
  }

  const char* filename = argv[1];
  int SRC = std::atoi(argv[2]);
  int DST = std::atoi(argv[3]);

  // ─── Read graph header + edges on rank 0 ───────────────────────────────
  uint32_t N = 0, M = 0;
  std::vector<uint32_t> flat;
  if (rank == 0) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
      std::cerr << "Cannot open " << filename << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    in.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&M), sizeof(uint32_t));
    flat.resize(2 * M);
    in.read(reinterpret_cast<char*>(flat.data()), flat.size() * sizeof(uint32_t));
  }
  // ─── Broadcast N, M, then flat edges to all ranks ───────────────────────
  MPI_Bcast(&N, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&M, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  if (rank != 0) flat.resize(2 * M);
  MPI_Bcast(flat.data(), flat.size(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // ─── Build CSR on every rank ────────────────────────────────────────────
  std::vector<int> deg(N, 0);
  for (size_t i = 0; i < flat.size(); i += 2) {
    deg[flat[i]]++;
  }
  std::vector<int> row_ptr(N+1, 0);
  for (int i = 0; i < (int)N; i++) {
    row_ptr[i+1] = row_ptr[i] + deg[i];
  }
  std::vector<int> col_ind(2*M), cur = row_ptr;
  for (size_t i = 0; i < flat.size(); i += 2) {
    int u = flat[i], v = flat[i+1];
    col_ind[cur[u]++] = v;
  }

  // ─── Initialize host-side BFS arrays ──────────────────────────────────
  std::vector<int> h_vis_s(N,0), h_vis_t(N,0);
  std::vector<int> h_front_s(N,0), h_front_t(N,0);
  std::vector<int> h_pred_s(N,-1), h_pred_t(N,-1);
  h_vis_s[SRC] = 1; h_front_s[SRC] = 1; h_pred_s[SRC] = SRC;
  h_vis_t[DST] = 1; h_front_t[DST] = 1; h_pred_t[DST] = DST;

  int threads = 256;
  int blocks  = (N + threads - 1) / threads;

  // ─── Hand off to GPU setup ─────────────────────────────────────────────
  gpuSetup(
    N, 2*M,
    row_ptr.data(), col_ind.data(),
    SRC, DST,
    threads, blocks,
    h_vis_s.data(), h_vis_t.data(),
    h_front_s.data(), h_front_t.data(),
    h_pred_s.data(), h_pred_t.data()
  );

  // ─── Time the bidirectional BFS ────────────────────────────────────────
  MPI_Barrier(MPI_COMM_WORLD);
  double t_start = MPI_Wtime();

  bool global_hit = false;
  while (!global_hit) {
    unsigned char schg, tchg, inter;
    gpuIterate(&schg, &tchg, &inter);

    // 1) Merge visited/frontiers/preds across ranks
    MPI_Allreduce(MPI_IN_PLACE, h_vis_s.data(),   N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_vis_t.data(),   N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_front_s.data(), N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_front_t.data(), N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_pred_s.data(),  N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, h_pred_t.data(),  N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // 2) Copy merged state back to device
    gpuCopyHostToDevice();

    // 3) Check for intersection (global)
    int local_hit = inter ? 1 : 0;
    MPI_Allreduce(&local_hit, &global_hit, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (global_hit) break;

    // 4) Check if any rank made progress
    int local_change = (schg || tchg) ? 1 : 0;
    int global_change;
    MPI_Allreduce(&local_change, &global_change, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (!global_change) break;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t_end = MPI_Wtime();

  // ─── Rank 0 prints results ─────────────────────────────────────────────
  if (rank == 0) {
    std::cout << "Running with " << size << " ranks\n";
    std::cout << "Search time: " << (t_end - t_start) << " seconds\n";

    int meet = -1;
    for (int i = 0; i < (int)h_vis_s.size(); i++) {
      if (h_vis_s[i] && h_vis_t[i]) { meet = i; break; }
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
      for (auto v : path) std::cout << " " << v;
      std::cout << "\n";
    }
  }

  gpuFinalize();
  MPI_Finalize();
  return 0;
}
