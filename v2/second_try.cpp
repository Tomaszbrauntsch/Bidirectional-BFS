// mpi_bibfs_bitset.cpp
#include <mpi.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

// popcount on 64‑bit word
static inline int popcnt(uint64_t x) {
    return __builtin_popcountll(x);
}

int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    if(argc!=4){
        if(rank==0) 
            std::cerr<<"Usage: mpirun -n <p> ./mpi_bibfs_bitset <graph.bin> <src> <dst>\n";
        MPI_Finalize();
        return 1;
    }

    const char* filename = argv[1];
    int src = std::atoi(argv[2]);
    int dst = std::atoi(argv[3]);

    // ─── Phase 1: load & build adjacency list ───────────────────────────────
    uint32_t n, m;
    std::vector<uint32_t> flat_edges;
    if(rank==0){
        std::ifstream in(filename, std::ios::binary);
        if(!in){ std::cerr<<"Cannot open "<<filename<<"\n"; MPI_Abort(MPI_COMM_WORLD,1); }
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        in.read(reinterpret_cast<char*>(&m), sizeof(m));
        flat_edges.resize(2*m);
        in.read(reinterpret_cast<char*>(flat_edges.data()), 2*m*sizeof(uint32_t));
        if(!in){ std::cerr<<"Unexpected EOF\n"; MPI_Abort(MPI_COMM_WORLD,1); }
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if(rank!=0) flat_edges.resize(2*m);
    MPI_Bcast(flat_edges.data(), 2*m, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::vector<std::vector<int>> adj(n);
    for(uint32_t i=0; i<2*m; i+=2){
        int u = flat_edges[i], v = flat_edges[i+1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ─── Phase 2: bidirectional bitset BFS ─────────────────────────────────
    int L = (n + 63) >> 6;  // number of 64‑bit words to cover n bits
    std::vector<uint64_t>
      frontierSrc(L,0), frontierDst(L,0),
      nextF(L,0),
      visitedSrc(L,0), visitedDst(L,0);

    // initialize both frontiers & visited
    frontierSrc[src>>6] |= 1ULL << (src & 63);
    visitedSrc [src>>6] |= 1ULL << (src & 63);
    frontierDst[dst>>6] |= 1ULL << (dst & 63);
    visitedDst [dst>>6] |= 1ULL << (dst & 63);

    int distance = 0;
    bool found = false;

    double t0 = MPI_Wtime();
    while(!found){
        // 1) pick the smaller frontier
        int cntS = 0, cntD = 0;
        for(int i=0;i<L;i++){
            cntS += popcnt(frontierSrc[i]);
            cntD += popcnt(frontierDst[i]);
        }
        bool expandSrc = (cntS <= cntD);

        // 2) local expansion of the chosen frontier
        for(uint32_t u = rank; u < n; u += size){
            uint64_t mask_u = 1ULL << (u & 63);
            if(expandSrc) {
                if(frontierSrc[u>>6] & mask_u) {
                    for(int v: adj[u]){
                        uint64_t m_v = 1ULL << (v & 63);
                        if(!(visitedSrc[v>>6] & m_v)){
                            visitedSrc[v>>6] |= m_v;
                            nextF      [v>>6] |= m_v;
                        }
                    }
                }
            } else {
                if(frontierDst[u>>6] & mask_u) {
                    for(int v: adj[u]){
                        uint64_t m_v = 1ULL << (v & 63);
                        if(!(visitedDst[v>>6] & m_v)){
                            visitedDst[v>>6] |= m_v;
                            nextF      [v>>6] |= m_v;
                        }
                    }
                }
            }
        }

        // 3) sync bitsets (one OR‐reduction for next frontier,
        //    one for visited)
        MPI_Allreduce(MPI_IN_PLACE, nextF.data(), L,
                      MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
        if(expandSrc) {
            MPI_Allreduce(MPI_IN_PLACE, visitedSrc.data(), L,
                          MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
            frontierSrc.swap(nextF);
        } else {
            MPI_Allreduce(MPI_IN_PLACE, visitedDst.data(), L,
                          MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
            frontierDst.swap(nextF);
        }
        std::fill(nextF.begin(), nextF.end(), 0ULL);

        distance++;

        // 4) check for meeting: does any bit overlap?
        bool local_hit = false;
        for(int i=0;i<L;i++){
            if(visitedSrc[i] & visitedDst[i]){
                local_hit = true;
                break;
            }
        }
        int hit = local_hit ? 1 : 0, global_hit;
        MPI_Allreduce(&hit, &global_hit, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if(global_hit) found = true;
    }
    double t1 = MPI_Wtime();

    // ─── Rank 0: report ─────────────────────────────────────────────────────
    if(rank==0){
        std::cout<<"Shortest path length = "<<distance<<"\n";
        std::cout<<"[Time] bidir‐bitset BFS = "<<(t1-t0)<<" seconds\n";
    }

    MPI_Finalize();
    return 0;
}
