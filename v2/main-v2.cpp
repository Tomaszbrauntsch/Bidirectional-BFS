#include <mpi.h>
#include <iostream>
#include <vector>
#include <climits>
#include <algorithm>
using namespace std;

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n = 6;
    vector<vector<int>> graph = {
        {0, 1, 0, 0, 1, 0},
        {1, 0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0, 0},
        {0, 0, 1, 0, 1, 1},
        {1, 1, 0, 1, 0, 0},
        {0, 0, 0, 1, 0, 0}
    };

    int src = 3, dest = 5;
    if(src == dest){
        if(rank==0) cout << "Shortest path length is 0" << endl;
        MPI_Finalize();
        return 0;
    }
    vector<int> visitedSrc(n, 0), visitedDest(n, 0);
    vector<int> distSrc(n, INT_MAX), distDest(n, INT_MAX);
    
    visitedSrc[src] = 1;
    distSrc[src] = 0;
    visitedDest[dest] = 1;
    distDest[dest] = 0;
    vector<int> frontierSrc = { src };
    vector<int> frontierDest = { dest };
    
    int candidateAnswer = INT_MAX;
    bool done = false;
    
    while(!frontierSrc.empty() && !frontierDest.empty() && !done){
        vector<int> localNewFrontierSrc;
        for (int i = rank; i < frontierSrc.size(); i += size) {
            int u = frontierSrc[i];
            for (int v = 0; v < n; v++) {
                if (graph[u][v] && !visitedSrc[v]) {
                    visitedSrc[v] = 1;
                    distSrc[v] = distSrc[u] + 1;
                    localNewFrontierSrc.push_back(v);
                }
            }
        }
        int localCount = localNewFrontierSrc.size();
        vector<int> counts(size, 0);
        MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int totalNew = 0;
        vector<int> displs(size, 0);
        for (int i = 0; i < size; i++) {
            displs[i] = totalNew;
            totalNew += counts[i];
        }
        vector<int> globalNewFrontierSrc(totalNew);
        MPI_Allgatherv(localNewFrontierSrc.data(), localCount, MPI_INT,
                       globalNewFrontierSrc.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
        frontierSrc = globalNewFrontierSrc;
        MPI_Allreduce(MPI_IN_PLACE, visitedSrc.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, distSrc.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        bool localFound = false;
        for (int i = 0; i < n; i++) {
            if (visitedSrc[i] && visitedDest[i]) {
                int candidate = distSrc[i] + distDest[i];
                candidateAnswer = min(candidateAnswer, candidate);
                localFound = true;
            }
        }
        int globalFound;
        int localFoundInt = localFound ? 1 : 0;
        MPI_Allreduce(&localFoundInt, &globalFound, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &candidateAnswer, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (globalFound){
            done = true;
            break;
        }
        vector<int> localNewFrontierDest;
        for (int i = rank; i < frontierDest.size(); i += size) {
            int u = frontierDest[i];
            for (int v = 0; v < n; v++) {
                if (graph[u][v] && !visitedDest[v]) {
                    visitedDest[v] = 1;
                    distDest[v] = distDest[u] + 1;
                    localNewFrontierDest.push_back(v);
                }
            }
        }
        localCount = localNewFrontierDest.size();
        counts.assign(size, 0);
        MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        totalNew = 0;
        displs.assign(size, 0);
        for (int i = 0; i < size; i++) {
            displs[i] = totalNew;
            totalNew += counts[i];
        }
        vector<int> globalNewFrontierDest(totalNew);
        MPI_Allgatherv(localNewFrontierDest.data(), localCount, MPI_INT,
                       globalNewFrontierDest.data(), counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
        frontierDest = globalNewFrontierDest;
        
        MPI_Allreduce(MPI_IN_PLACE, visitedDest.data(), n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, distDest.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        localFound = false;
        for (int i = 0; i < n; i++) {
            if (visitedSrc[i] && visitedDest[i]) {
                int candidate = distSrc[i] + distDest[i];
                candidateAnswer = min(candidateAnswer, candidate);
                localFound = true;
            }
        }
        localFoundInt = localFound ? 1 : 0;
        MPI_Allreduce(&localFoundInt, &globalFound, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &candidateAnswer, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (globalFound){
            done = true;
            break;
        }
    }
    
    int finalCandidate;
    MPI_Allreduce(&candidateAnswer, &finalCandidate, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        if(finalCandidate == INT_MAX)
            cout << "No path found between " << src << " and " << dest << endl;
        else
            cout << "Shortest path length from " << src << " to " << dest << " is: " << finalCandidate << endl;
    }
    
    MPI_Finalize();
    return 0;
}
