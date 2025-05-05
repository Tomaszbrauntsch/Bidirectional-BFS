#include "comp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// simple CUDA error checker
#define checkCuda(err)                                                     \
  if ((err) != cudaSuccess) {                                              \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
            cudaGetErrorString(err));                                      \
    exit(-1);                                                               \
  }

// device pointers
static int *d_row_ptr, *d_col_ind;
static int *d_vis_s, *d_vis_t;
static int *d_front_s, *d_front_t;
static int *d_front_next;
static unsigned char *d_changed_s, *d_changed_t, *d_intersect;

// expand one level of BFS on whichever side
__global__ void expand_frontier_kernel(
    int N, int rank, int size,
    const int *row_ptr, const int *col_ind,
    const int *frontier_in, int *frontier_out,
    int *visited, unsigned char *changed)
{
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  // strided partition
  if (u >= N || (u % size) != rank || frontier_in[u] == 0) return;
  int start = row_ptr[u], end = row_ptr[u+1];
  for (int e = start; e < end; ++e) {
    int v = col_ind[e];
    if (visited[v] == 0) {
      if (atomicExch(&visited[v], 1) == 0) {
        frontier_out[v] = 1;
        *changed = 1;
      }
    }
  }
}

// detect any overlap in visited arrays
__global__ void check_intersect_kernel(
    int N, const int *vis1, const int *vis2, unsigned char *found)
{
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u < N && vis1[u] && vis2[u]) {
    *found = 1;
  }
}

void cudaInitGraph(int N, int M,
                   const int *h_row_ptr,
                   const int *h_col_ind)
{
  size_t rp_bytes = (N+1) * sizeof(int),
         ci_bytes = (2*M) * sizeof(int),
         nv_bytes = N * sizeof(int);

  // copy CSR
  checkCuda(cudaMalloc(&d_row_ptr, rp_bytes));
  checkCuda(cudaMalloc(&d_col_ind, ci_bytes));
  checkCuda(cudaMemcpy(d_row_ptr, h_row_ptr, rp_bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_col_ind, h_col_ind, ci_bytes, cudaMemcpyHostToDevice));

  // allocate int arrays for BFS
  checkCuda(cudaMalloc(&d_vis_s, nv_bytes));
  checkCuda(cudaMalloc(&d_vis_t, nv_bytes));
  checkCuda(cudaMalloc(&d_front_s, nv_bytes));
  checkCuda(cudaMalloc(&d_front_t, nv_bytes));
  checkCuda(cudaMalloc(&d_front_next, nv_bytes));

  // allocate single-byte flags
  checkCuda(cudaMalloc(&d_changed_s, 1));
  checkCuda(cudaMalloc(&d_changed_t, 1));
  checkCuda(cudaMalloc(&d_intersect, 1));

  // zero‐initialize
  checkCuda(cudaMemset(d_vis_s, 0, nv_bytes));
  checkCuda(cudaMemset(d_vis_t, 0, nv_bytes));
  checkCuda(cudaMemset(d_front_s, 0, nv_bytes));
  checkCuda(cudaMemset(d_front_t, 0, nv_bytes));
}

void cudaInitFrontiers(int src, int dst)
{
  int one_i = 1;
  checkCuda(cudaMemcpy(d_vis_s + src,   &one_i, sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_front_s + src, &one_i, sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_vis_t + dst,   &one_i, sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_front_t + dst, &one_i, sizeof(int), cudaMemcpyHostToDevice));
}

void cudaExpandFrontier(int side,
                        int *h_front_in,
                        int *h_front_out,
                        int *h_vis,
                        unsigned char *h_changed,
                        int N)
{
  int *d_front   = (side == 0 ? d_front_s   : d_front_t);
  int *d_visited = (side == 0 ? d_vis_s     : d_vis_t);
  unsigned char *d_changed = (side == 0 ? d_changed_s : d_changed_t);

  // upload host frontier & clear device buffers
  checkCuda(cudaMemcpy(d_front,        h_front_in,  N * sizeof(int),       cudaMemcpyHostToDevice));
  checkCuda(cudaMemset(d_front_next,   0,           N * sizeof(int)));
  checkCuda(cudaMemset(d_changed,      0,           1));

  // launch kernel (rank=0,size=1 here; MPI driver overrides if needed)
  int threads = 256;
  int blocks  = (N + threads - 1) / threads;
  expand_frontier_kernel<<<blocks, threads>>>(N, 0, 1,
                                              d_row_ptr, d_col_ind,
                                              d_front, d_front_next,
                                              d_visited, d_changed);
  checkCuda(cudaDeviceSynchronize());

  // download results
  checkCuda(cudaMemcpy(h_front_out, d_front_next, N * sizeof(int),          cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_vis,       d_visited,    N * sizeof(int),          cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_changed,   d_changed,    sizeof(unsigned char),    cudaMemcpyDeviceToHost));
}

void cudaCheckIntersect(int *h_vis_s,
                        int *h_vis_t,
                        unsigned char *h_found,
                        int N)
{
  checkCuda(cudaMemset(d_intersect, 0, 1));
  int threads = 256, blocks = (N + threads - 1) / threads;
  check_intersect_kernel<<<blocks, threads>>>(N, d_vis_s, d_vis_t, d_intersect);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(h_found, d_intersect, 1, cudaMemcpyDeviceToHost));
}

void cudaFreeGraph()
{
  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_vis_s);
  cudaFree(d_vis_t);
  cudaFree(d_front_s);
  cudaFree(d_front_t);
  cudaFree(d_front_next);
  cudaFree(d_changed_s);
  cudaFree(d_changed_t);
  cudaFree(d_intersect);
}
