#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda.h>

#define checkCuda(err) \
    if((err)!=cudaSuccess){ \
        fprintf(stderr,"CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(-1); \
    }

// expand one bfs frontier using 32-bit arrays
    __global__ void expand_frontier(
    int N, const int *row_ptr, const int *col_ind,
    const int *frontier_in, int *frontier_out,
    int *visited, unsigned char *changed
    ) {
    int tid_in_block = threadIdx.x;
    int bid = blockIdx.x;
    int u = bid * blockDim.x + tid_in_block;

    // log entry for every thread
    //printf("[expand_frontier] block %2d thread %3d → global %4d\n", bid, tid_in_block, u);

    if(u >= N || frontier_in[u] == 0) {
        //printf("[expand_frontier] global %4d → skipping (out‐of‐range or not in frontier)\n", u);
        return;
    }

    //printf("[expand_frontier] Thread %4d: expanding node %4d\n", u, u);
    int start = row_ptr[u];
    int end   = row_ptr[u+1];
    for(int e = start; e < end; ++e) {
        int v = col_ind[e];
        // atomically set visited[v] to 1, old value indicates whether it was new
        int old = atomicExch(&visited[v], 1);
        if(old == 0) {
            frontier_out[v] = 1;
            *changed = 1;
            //printf("[expand_frontier] Thread %4d: discovered neighbor %4d\n", u, v);
        }
    }
}

    __global__ void check_intersect(
        int            N,
        const int     *vis1,
        const int     *vis2,
        unsigned char *found
    ) {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int u   = bid * blockDim.x + tid;

        // log entry for every thread
        //printf("[check_intersect] block %2d thread %3d → checking node %4d\n", bid, tid, u);

        if(u < N && vis1[u] && vis2[u]){
            //printf("[check_intersect] Thread %4d: intersection FOUND at node %4d\n", u, u);
            *found = 1;
        }
}

int main(int argc, char** argv){
    if(argc < 4){
        printf("Usage: %s <bin-file> <src> <dst>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int SRC = atoi(argv[2]);
    int DST = atoi(argv[3]);
    FILE* f = fopen(path, "rb");
    if(!f){ perror("fopen"); return 1; }
    unsigned int N, M;
    fread(&N, sizeof(N), 1, f);
    fread(&M, sizeof(M), 1, f);

    std::vector<std::pair<int,int>> edges;
    edges.reserve(M * 2);
    for(unsigned int i = 0; i < M; i++){
        unsigned int u, v;
        fread(&u, sizeof(u), 1, f);
        fread(&v, sizeof(v), 1, f);
        edges.emplace_back(u, v);
        edges.emplace_back(v, u);
    }
    fclose(f);

    std::vector<int> deg(N, 0);
    for(auto &e : edges) deg[e.first]++;
    std::vector<int> row_ptr(N+1, 0);
    for(int i = 0; i < N; i++) row_ptr[i+1] = row_ptr[i] + deg[i];

    std::vector<int> col_ind(edges.size());
    std::vector<int> cursor = row_ptr;
    for(auto &e : edges){
        int u = e.first, v = e.second;
        col_ind[cursor[u]++] = v;
    }

    // debug to see nieghbors
    printf("DEBUG: Node %d neighbors:", SRC);
    for(int i = row_ptr[SRC]; i < row_ptr[SRC+1]; i++) printf(" %d", col_ind[i]);
    printf("\n");
    printf("DEBUG: Node %d neighbors:", DST);
    for(int i = row_ptr[DST]; i < row_ptr[DST+1]; i++) printf(" %d", col_ind[i]);
    printf("\n");

    std::vector<char> vis_cpu(N, 0);
    std::vector<int> queue;
    queue.reserve(N);
    queue.push_back(SRC);
    vis_cpu[SRC] = 1;
    bool cpu_found = false;
    for(size_t qi = 0; qi < queue.size(); qi++){
        int u = queue[qi];
        if(u == DST){ cpu_found = true; break; }
        for(int e = row_ptr[u]; e < row_ptr[u+1]; e++){
            int v = col_ind[e];
            if(!vis_cpu[v]){
                vis_cpu[v] = 1;
                queue.push_back(v);
            }
        }
    }
    printf("DEBUG: CPU BFS found? %s, visited %zu nodes\n", cpu_found?"YES":"NO", queue.size());

    edges.clear(); deg.clear(); cursor.clear();
    int *d_row_ptr, *d_col_ind;
    checkCuda(cudaMalloc(&d_row_ptr, (N+1) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_ind, col_ind.size() * sizeof(int)));
    checkCuda(cudaMemcpy(d_row_ptr, row_ptr.data(), (N+1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_ind, col_ind.data(), col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));

    // NEED TO ALLOCATE IN 32 BITS OTHERWISE IT CRASHES
    int *d_vis_s, *d_vis_t, *d_front_s, *d_front_t, *d_front_next;
    unsigned char *d_changed_s, *d_changed_t, *d_intersect;
    checkCuda(cudaMalloc(&d_vis_s,      N * sizeof(int)));
    checkCuda(cudaMalloc(&d_vis_t,      N * sizeof(int)));
    checkCuda(cudaMalloc(&d_front_s,    N * sizeof(int)));
    checkCuda(cudaMalloc(&d_front_t,    N * sizeof(int)));
    checkCuda(cudaMalloc(&d_front_next, N * sizeof(int)));
    checkCuda(cudaMalloc(&d_changed_s,  sizeof(unsigned char)));
    checkCuda(cudaMalloc(&d_changed_t,  sizeof(unsigned char)));
    checkCuda(cudaMalloc(&d_intersect,  sizeof(unsigned char)));

    cudaMemset(d_vis_s,      0, N * sizeof(int));
    cudaMemset(d_vis_t,      0, N * sizeof(int));
    cudaMemset(d_front_s,    0, N * sizeof(int));
    cudaMemset(d_front_t,    0, N * sizeof(int));
    cudaMemset(d_front_next, 0, N * sizeof(int));
    cudaMemset(d_intersect,  0, sizeof(unsigned char));
    int one = 1;
    checkCuda(cudaMemcpy(d_vis_s + SRC,   &one, sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_front_s + SRC, &one, sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_vis_t + DST,   &one, sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_front_t + DST, &one, sizeof(int), cudaMemcpyHostToDevice));
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    bool found = false;
    // std::vector<int> f_s(N), f_t(N);
    // cudaMemcpy(f_s.data(), d_front_s, N * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(f_t.data(), d_front_t, N * sizeof(int), cudaMemcpyDeviceToHost);
    // int c_s = std::count(f_s.begin(), f_s.end(), 1);
    // int c_t = std::count(f_t.begin(), f_t.end(), 1);
    // printf("DEBUG: Initial frontier_s count = %d, frontier_t count = %d\n", c_s, c_t);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start));
    // alternate expanisons because only way I could make it work
    for(int iter = 0; iter < N && !found; iter++){
        unsigned char zero = 0, schg = 0, tchg = 0, inter = 0;

        cudaMemcpy(d_changed_s, &zero, sizeof(zero), cudaMemcpyHostToDevice);
        cudaMemset(d_front_next, 0, N * sizeof(int));
        expand_frontier<<<blocks, threads>>>(N, d_row_ptr, d_col_ind, d_front_s, d_front_next, d_vis_s, d_changed_s);
        cudaError_t e = cudaGetLastError();
        if(e != cudaSuccess)  
            printf("Kernel launch error: %s\n", cudaGetErrorString(e));
        checkCuda(cudaDeviceSynchronize());
        cudaMemcpy(&schg, d_changed_s, sizeof(schg), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_front_s, d_front_next, N * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_changed_t, &zero, sizeof(zero), cudaMemcpyHostToDevice);
        cudaMemset(d_front_next, 0, N * sizeof(int));
        expand_frontier<<<blocks, threads>>>(N, d_row_ptr, d_col_ind, d_front_t, d_front_next, d_vis_t, d_changed_t);
        e = cudaGetLastError();
        if(e != cudaSuccess)  
            printf("Kernel launch error: %s\n", cudaGetErrorString(e));
        checkCuda(cudaDeviceSynchronize());
        cudaMemcpy(&tchg, d_changed_t, sizeof(tchg), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_front_t, d_front_next, N * sizeof(int), cudaMemcpyDeviceToDevice);
        printf("Iter %d: schg=%d, tchg=%d\n", iter, (int)schg, (int)tchg);
        cudaMemcpy(d_intersect, &zero, sizeof(zero), cudaMemcpyHostToDevice);
        check_intersect<<<blocks, threads>>>(N, d_vis_s, d_vis_t, d_intersect);
        checkCuda(cudaDeviceSynchronize());
        cudaMemcpy(&inter, d_intersect, sizeof(inter), cudaMemcpyDeviceToHost);
        if(inter){ found = true; break; }

        // ggs
        if(!schg && !tchg) break;
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    printf("DEBUG: gpu bfs time = %f ms\n", ms);

    std::vector<int> vs(N), vt(N);
    cudaMemcpy(vs.data(), d_vis_s, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vt.data(), d_vis_t, N*sizeof(int), cudaMemcpyDeviceToHost);
    int count_s=0, count_t=0;
    for(int i=0;i<N;i++){
      count_s += (vs[i]!=0);
      count_t += (vt[i]!=0);
    }
    printf("DEBUG: GPU visited_s count = %d\n", count_s);
    printf("DEBUG: GPU visited_t count = %d\n", count_t);
    if(vs[DST]) printf("DEBUG: source‐side actually reached DST on GPU!\n");
    if(vt[SRC]) printf("DEBUG: target‐side actually reached SRC on GPU!\n");


    printf(found ? "GPU BFS: PATH FOUND\n" : "GPU BFS: NO PATH\n");

    // cleanup
    cudaFree(d_row_ptr);   cudaFree(d_col_ind);
    cudaFree(d_vis_s);     cudaFree(d_vis_t);
    cudaFree(d_front_s);   cudaFree(d_front_t);
    cudaFree(d_front_next);
    cudaFree(d_changed_s); cudaFree(d_changed_t);
    cudaFree(d_intersect);
    return 0;
}
