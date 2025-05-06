// comp.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

// Error checking macro
#define checkCuda(err) \
  if ((err) != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(-1); \
  }

// Device‐global data
static int   N_glob;
static int   threads_glob, blocks_glob;
static int  *d_row_ptr, *d_col_ind;
static int  *d_vis_s, *d_vis_t, *d_front_s, *d_front_t, *d_front_next;
static int  *d_pred_s, *d_pred_t;
static unsigned char *d_changed_s, *d_changed_t, *d_intersect;
// Host pointers (set by gpuSetup)
static int *h_vis_s, *h_vis_t, *h_front_s, *h_front_t, *h_pred_s, *h_pred_t;

// Kernel: expand one frontier, record predecessors
__global__ void expand_frontier(
  int N,
  const int *row_ptr,
  const int *col_ind,
  const int *frontier_in,
        int *frontier_out,
        int *visited,
        unsigned char *changed,
        int *pred
) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= N || frontier_in[u] == 0) return;
  int start = row_ptr[u], end = row_ptr[u+1];
  for (int e = start; e < end; ++e) {
    int v = col_ind[e];
    int old = atomicExch(&visited[v], 1);
    if (old == 0) {
      frontier_out[v] = 1;
      *changed        = 1;
      pred[v]         = u;
    }
  }
}

// Kernel: detect intersection
__global__ void check_intersect(
  int N,
  const int *vis1,
  const int *vis2,
        unsigned char *found
) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u < N && vis1[u] && vis2[u]) *found = 1;
}

extern "C" void gpuSetup(
  int N, int M,
  int *h_row_ptr, int *h_col_ind,
  int SRC, int DST,
  int threads, int blocks,
  int *hv_s, int *hv_t,
  int *hf_s, int *hf_t,
  int *hp_s, int *hp_t
) {
  N_glob       = N;
  threads_glob = threads;
  blocks_glob  = blocks;
  h_vis_s   = hv_s;
  h_vis_t   = hv_t;
  h_front_s = hf_s;
  h_front_t = hf_t;
  h_pred_s  = hp_s;
  h_pred_t  = hp_t;

  // Allocate CSR on device
  checkCuda(cudaMalloc(&d_row_ptr, (N+1)*sizeof(int)));
  checkCuda(cudaMalloc(&d_col_ind, M    *sizeof(int)));
  checkCuda(cudaMemcpy(d_row_ptr, h_row_ptr, (N+1)*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_col_ind, h_col_ind, M    *sizeof(int), cudaMemcpyHostToDevice));

  // Allocate BFS arrays
  #define ALLOC_INT(ptr) checkCuda(cudaMalloc(&ptr, N*sizeof(int)))
  ALLOC_INT(d_vis_s);
  ALLOC_INT(d_vis_t);
  ALLOC_INT(d_front_s);
  ALLOC_INT(d_front_t);
  ALLOC_INT(d_front_next);
  ALLOC_INT(d_pred_s);
  ALLOC_INT(d_pred_t);
  #undef ALLOC_INT
  checkCuda(cudaMalloc(&d_changed_s,  sizeof(unsigned char)));
  checkCuda(cudaMalloc(&d_changed_t,  sizeof(unsigned char)));
  checkCuda(cudaMalloc(&d_intersect,  sizeof(unsigned char)));

  // Initialize device arrays from host
  checkCuda(cudaMemcpy(d_vis_s,   h_vis_s,   N*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_vis_t,   h_vis_t,   N*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_front_s, h_front_s, N*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_front_t, h_front_t, N*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_pred_s,  h_pred_s,  N*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_pred_t,  h_pred_t,  N*sizeof(int), cudaMemcpyHostToDevice));
}

extern "C" void gpuIterate(
  unsigned char *schg,
  unsigned char *tchg,
  unsigned char *inter
) {
  unsigned char zero = 0;
  // Reset change flags
  checkCuda(cudaMemcpy(d_changed_s, &zero, 1, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_changed_t, &zero, 1, cudaMemcpyHostToDevice));

  // Expand source side
  checkCuda(cudaMemset(d_front_next, 0, N_glob*sizeof(int)));
  expand_frontier<<<blocks_glob,threads_glob>>>(
    N_glob, d_row_ptr, d_col_ind,
    d_front_s, d_front_next,
    d_vis_s,   d_changed_s,
    d_pred_s
  );
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(d_front_s, d_front_next, N_glob*sizeof(int), cudaMemcpyDeviceToDevice));

  // Expand target side
  checkCuda(cudaMemset(d_front_next, 0, N_glob*sizeof(int)));
  expand_frontier<<<blocks_glob,threads_glob>>>(
    N_glob, d_row_ptr, d_col_ind,
    d_front_t, d_front_next,
    d_vis_t,   d_changed_t,
    d_pred_t
  );
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(d_front_t, d_front_next, N_glob*sizeof(int), cudaMemcpyDeviceToDevice));

  // Copy device → host
  checkCuda(cudaMemcpy(h_vis_s,   d_vis_s,   N_glob*sizeof(int), cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_vis_t,   d_vis_t,   N_glob*sizeof(int), cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_front_s, d_front_s, N_glob*sizeof(int), cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_front_t, d_front_t, N_glob*sizeof(int), cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_pred_s,  d_pred_s,  N_glob*sizeof(int), cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_pred_t,  d_pred_t,  N_glob*sizeof(int), cudaMemcpyDeviceToHost));

  // Copy change flags
  checkCuda(cudaMemcpy(schg, d_changed_s, 1, cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(tchg, d_changed_t, 1, cudaMemcpyDeviceToHost));

  // Detect intersection
  checkCuda(cudaMemset(d_intersect, 0, 1));
  check_intersect<<<blocks_glob,threads_glob>>>(N_glob, d_vis_s, d_vis_t, d_intersect);
  checkCuda(cudaDeviceSynchronize());
  checkCuda(cudaMemcpy(inter, d_intersect, 1, cudaMemcpyDeviceToHost));
}

extern "C" void gpuCopyHostToDevice() {
  checkCuda(cudaMemcpy(d_vis_s,   h_vis_s,   N_glob*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_vis_t,   h_vis_t,   N_glob*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_front_s, h_front_s, N_glob*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_front_t, h_front_t, N_glob*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_pred_s,  h_pred_s,  N_glob*sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_pred_t,  h_pred_t,  N_glob*sizeof(int), cudaMemcpyHostToDevice));
}

extern "C" void gpuFinalize() {
  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_vis_s);
  cudaFree(d_vis_t);
  cudaFree(d_front_s);
  cudaFree(d_front_t);
  cudaFree(d_front_next);
  cudaFree(d_pred_s);
  cudaFree(d_pred_t);
  cudaFree(d_changed_s);
  cudaFree(d_changed_t);
  cudaFree(d_intersect);
}
