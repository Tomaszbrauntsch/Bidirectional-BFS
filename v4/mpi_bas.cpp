// mpi_bas.cpp
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include "comp.h"

int main(int argc, char* argv[]){
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
  int src = std::atoi(argv[2]);
  int dst = std::atoi(argv[3]);

  // 1) Read & broadcast CSR and edge‐list
  uint32_t N, M;
  std::vector<uint32_t> flat;
  if (rank == 0) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr<<"Cannot open "<<filename<<"\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    in.read(reinterpret_cast<char*>(&N), sizeof(N));
    in.read(reinterpret_cast<char*>(&M), sizeof(M));
    flat.resize(2*M);
    in.read(reinterpret_cast<char*>(flat.data()), 2*M*sizeof(uint32_t));
  }
  MPI_Bcast(&N, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&M, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  if (rank != 0) flat.resize(2*M);
  MPI_Bcast(flat.data(), 2*M, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // 2) Build CSR host‐side
  std::vector<int> deg(N,0), row_ptr(N+1,0), col_ind(2*M);
  for (size_t i = 0; i < 2*M; i += 2) {
    deg[flat[i]]++;
    deg[flat[i+1]]++;
  }
  for (size_t i = 0; i < N; i++) row_ptr[i+1] = row_ptr[i] + deg[i];
  std::vector<int> cur = row_ptr;
  for (size_t i = 0; i < 2*M; i += 2) {
    int u = flat[i], v = flat[i+1];
    col_ind[cur[u]++] = v;
    col_ind[cur[v]++] = u;
  }

  // 2b) ALSO build an adjacency list for host‐side path recovery
  std::vector<std::vector<int>> adj(N);
  for (size_t i = 0; i < 2*M; i += 2) {
    int u = flat[i], v = flat[i+1];
    adj[u].push_back(v);
    adj[v].push_back(u);
  }

  // 3) Upload graph + seed both frontiers on GPU
  cudaInitGraph(N, M, row_ptr.data(), col_ind.data());
  cudaInitFrontiers(src, dst);

  // 4) Host buffers (int!)
  std::vector<int> front_s(N,0), front_t(N,0), nextF(N,0), vis_s(N,0), vis_t(N,0);
  front_s[src] = vis_s[src] = 1;
  front_t[dst] = vis_t[dst] = 1;

  int distance = 0;
  bool found = false;
  double t0 = MPI_Wtime();

  // 5) Bidirectional BFS loop
  while (!found) {
    // a) local intersection test
    for (int i = 0; i < (int)N; i++) {
      if (vis_s[i] && vis_t[i]) { found = true; break; }
    }
    int hit = found ? 1 : 0, global_hit;
    MPI_Allreduce(&hit, &global_hit, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    found = global_hit;
    if (found) break;

    // b) pick smaller frontier
    int cntS = std::count(front_s.begin(), front_s.end(), 1);
    int cntT = std::count(front_t.begin(), front_t.end(), 1);
    bool expandSrc = (cntS <= cntT);

    // c) expand on GPU
    unsigned char changed = 0;
    if (expandSrc) {
      cudaExpandFrontier(0, front_s.data(), nextF.data(), vis_s.data(), &changed, N);
    } else {
      cudaExpandFrontier(1, front_t.data(), nextF.data(), vis_t.data(), &changed, N);
    }

    // d) merge nextF across ranks
    MPI_Allreduce(MPI_IN_PLACE, nextF.data(), N, MPI_INT, MPI_BOR, MPI_COMM_WORLD);

    // e) update visited & frontier arrays
    if (expandSrc) {
      for (int i = 0; i < (int)N; i++)
        if (nextF[i] && !vis_s[i]) vis_s[i] = 1;
      front_s.swap(nextF);
    } else {
      for (int i = 0; i < (int)N; i++)
        if (nextF[i] && !vis_t[i]) vis_t[i] = 1;
      front_t.swap(nextF);
    }
    std::fill(nextF.begin(), nextF.end(), 0);

    // f) re‐seed GPU for next iteration
    cudaInitFrontiers(src, dst);

    distance++;
  }

  double t1 = MPI_Wtime();

  // 6) On rank 0, print distance, time, and then reconstruct & print the actual path
  if (rank == 0) {
    std::cout << "Shortest path length = " << distance << "\n"
              << "[Time] MPI+CUDA BI‑BFS = " << (t1 - t0) << " s\n";

    // Simple host BFS from src→dst to recover the path
    std::vector<int> parent(N, -1);
    std::queue<int> q;
    parent[src] = src;
    q.push(src);
    while (!q.empty()) {
      int u = q.front(); q.pop();
      if (u == dst) break;
      for (int v: adj[u]) {
        if (parent[v] == -1) {
          parent[v] = u;
          q.push(v);
        }
      }
    }

    if (parent[dst] == -1) {
      std::cout << "No path found (unexpected!)\n";
    } else {
      // Reconstruct path by walking parent[] backwards
      std::vector<int> path;
      for (int u = dst; u != src; u = parent[u])
        path.push_back(u);
      path.push_back(src);
      std::reverse(path.begin(), path.end());

      std::cout << "Path:";
      for (int u: path) std::cout << " " << u;
      std::cout << "\n";
    }
  }

  cudaFreeGraph();
  MPI_Finalize();
  return 0;
}
