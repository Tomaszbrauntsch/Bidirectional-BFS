// mpi_bibfs_bitset.cpp
#include <mpi.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <algorithm>

static inline int popcnt(uint64_t x) {
    return __builtin_popcountll(x);
}

int main(int argc, char* argv[]) {
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
    uint32_t n,m;
    std::vector<uint32_t> flat;
    if(rank==0){
        std::ifstream in(filename,std::ios::binary);
        if(!in){ MPI_Abort(MPI_COMM_WORLD,1); }
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        in.read(reinterpret_cast<char*>(&m), sizeof(m));
        flat.resize(2*m);
        in.read(reinterpret_cast<char*>(flat.data()), 2*m*sizeof(uint32_t));
    }
    MPI_Bcast(&n,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
    MPI_Bcast(&m,1,MPI_UNSIGNED,0,MPI_COMM_WORLD);
    if(rank!=0) flat.resize(2*m);
    MPI_Bcast(flat.data(),2*m,MPI_UNSIGNED,0,MPI_COMM_WORLD);
    std::vector<std::vector<int>> adj(n);
    for(uint32_t i=0;i<2*m;i+=2){
        int u = flat[i], v = flat[i+1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // init bitsets
    int L = (n+63)>>6;
    std::vector<uint64_t>
        frontierS(L,0), frontierT(L,0),
        nextS(L,0), nextT(L,0),
        visitedS(L,0), visitedT(L,0);

    frontierS[src>>6] |= 1ULL<<(src&63);
    visitedS [src>>6] |= 1ULL<<(src&63);
    frontierT[dst>>6] |= 1ULL<<(dst&63);
    visitedT [dst>>6] |= 1ULL<<(dst&63);

    int distance = 0;
    bool found = false;
    double t0 = MPI_Wtime();

    while(!found){
        // expand S
        std::fill(nextS.begin(), nextS.end(), 0ULL);
        for(uint32_t u=rank; u<(uint32_t)n; u+=size){
            if(frontierS[u>>6] & (1ULL<<(u&63))){
                for(int v: adj[u]){
                    uint64_t mv = 1ULL<<(v&63);
                    if(!(visitedS[v>>6]&mv)){
                        visitedS[v>>6] |= mv;
                        nextS[v>>6]    |= mv;
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, nextS.data(), L,
                      MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, visitedS.data(), L,
                      MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
        frontierS.swap(nextS);

        // expand T
        std::fill(nextT.begin(), nextT.end(), 0ULL);
        for(uint32_t u=rank; u<(uint32_t)n; u+=size){
            if(frontierT[u>>6] & (1ULL<<(u&63))){
                for(int v: adj[u]){
                    uint64_t mv = 1ULL<<(v&63);
                    if(!(visitedT[v>>6]&mv)){
                        visitedT[v>>6] |= mv;
                        nextT[v>>6]    |= mv;
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, nextT.data(), L,
                      MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, visitedT.data(), L,
                      MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD);
        frontierT.swap(nextT);

        distance++;

        // check intersect
        bool local_hit = false;
        for(int i=0;i<L;i++){
            if(visitedS[i] & visitedT[i]){ local_hit=true; break; }
        }
        int hit = local_hit?1:0, global_hit;
        MPI_Allreduce(&hit, &global_hit, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if(global_hit){ found=true; break; }
        int cntS=0,cntT=0;
        for(int i=0;i<L;i++){
            cntS += popcnt(frontierS[i]);
            cntT += popcnt(frontierT[i]);
        }
        int gS=0,gT=0;
        MPI_Allreduce(&cntS,&gS,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(&cntT,&gT,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        if(gS==0 && gT==0){
            if(rank==0) std::cout<<"No path found\n";
            break;
        }
    }

    double t1 = MPI_Wtime();
    if(rank==0){
        if(found)
            std::cout<<"Shortest path length = "<<distance<<"\n";
        std::cout<<"[Time] bidir‑bitset BFS = "<<(t1-t0)<<" seconds\n";
        // annoying to recreate
        std::vector<int> parent(n, -1);
        std::queue<int> q;
        parent[src] = src;
        q.push(src);
        while(!q.empty()){
            int u = q.front(); q.pop();
            if(u==dst) break;
            for(int v: adj[u]){
                if(parent[v]==-1){
                    parent[v] = u;
                    q.push(v);
                }
            }
        }

        if(parent[dst]==-1){
            std::cout<<"No path found (reconstruction)\n";
        } else {
            std::vector<int> path;
            for(int x=dst; x!=src; x=parent[x]) path.push_back(x);
            path.push_back(src);
            std::reverse(path.begin(), path.end());
            std::cout<<"Path:";
            for(int x: path) std::cout<<" "<<x;
            std::cout<<"\n";
        }
    }

    MPI_Finalize();
    return 0;
}
