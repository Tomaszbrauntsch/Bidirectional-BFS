#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Copy CSR graph (row_ptr: N+1, col_ind: 2*M) into GPU memory
void cudaInitGraph(int N, int M,
                   const int *h_row_ptr,
                   const int *h_col_ind);

// Seed the source and target frontiers/visited arrays on GPU
void cudaInitFrontiers(int src, int dst);

// Expand one BFS frontier on GPU:
//  side==0 → expand source side, side==1 → expand target side
//  h_front_in/out: host arrays of length N (int*)
//  h_vis: host visited array of length N (int*)
//  h_changed: single-byte flag on host (unsigned char*) set to 1 if any new node discovered
void cudaExpandFrontier(int side,
                        int *h_front_in,
                        int *h_front_out,
                        int *h_vis,
                        unsigned char *h_changed,
                        int N);

// (Optional) Check for any intersection on GPU
//  h_vis_s, h_vis_t: host visited arrays (int*)
//  h_found: single-byte host flag (unsigned char*) set to 1 if any overlap found
void cudaCheckIntersect(int *h_vis_s,
                        int *h_vis_t,
                        unsigned char *h_found,
                        int N);

// Free all GPU allocations
void cudaFreeGraph();

#ifdef __cplusplus
}
#endif
