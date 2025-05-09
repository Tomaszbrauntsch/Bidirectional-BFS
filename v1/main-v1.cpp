// this is the serial version
// https://zdimension.fr/everyone-gets-bidirectional-bfs-wrong/
// https://www.thealgorists.com/Algo/TwoEndBFS

// bibfs_serial.cpp
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <src> <dst>\n";
        return 1;
    }
    const char* filename = argv[1];
    int src = std::atoi(argv[2]);
    int dst = std::atoi(argv[3]);
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return 1;
    }
    uint32_t n, m;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&m), sizeof(m));
    std::vector<uint32_t> edges(2*m);
    in.read(reinterpret_cast<char*>(edges.data()), 2*m * sizeof(uint32_t));
    if (!in) {
        std::cerr << "Unexpected EOF reading edges\n";
        return 1;
    }
    std::vector<std::vector<int>> adj(n);
    for (uint32_t i = 0; i < 2*m; i += 2) {
        int u = edges[i], v = edges[i+1];
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    std::vector<char> visitedSrc(n, 0), visitedDst(n, 0);
    std::vector<int>  parentSrc(n, -1), parentDst(n, -1);
    std::vector<int>  frontierSrc, frontierDst, nextFrontier;

    visitedSrc[src] = 1; frontierSrc.push_back(src);
    visitedDst[dst] = 1; frontierDst.push_back(dst);

    int meet = -1;
    auto t0 = std::chrono::steady_clock::now();
    while (meet == -1 && !frontierSrc.empty() && !frontierDst.empty()) {
        bool expandSrc = frontierSrc.size() <= frontierDst.size();
        nextFrontier.clear();

        if (expandSrc) {
            for (int u : frontierSrc) {
                for (int v : adj[u]) {
                    if (!visitedSrc[v]) {
                        visitedSrc[v] = 1;
                        parentSrc[v]  = u;
                        nextFrontier.push_back(v);
                        if (visitedDst[v]) { meet = v; break; }
                    }
                }
                if (meet != -1) break;
            }
            frontierSrc.swap(nextFrontier);
        } else {
            for (int u : frontierDst) {
                for (int v : adj[u]) {
                    if (!visitedDst[v]) {
                        visitedDst[v] = 1;
                        parentDst[v]  = u;
                        nextFrontier.push_back(v);
                        if (visitedSrc[v]) { meet = v; break; }
                    }
                }
                if (meet != -1) break;
            }
            frontierDst.swap(nextFrontier);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    if (meet == -1) {
        std::cout << "No path found between " << src << " and " << dst << "\n";
    } else {
        std::vector<int> rev1;
        for (int cur = meet; cur != -1; cur = parentSrc[cur])
            rev1.push_back(cur);
        std::vector<int> rev2;
        for (int cur = meet; cur != -1; cur = parentDst[cur])
            rev2.push_back(cur);
        int hops = (int)rev1.size() - 1 + ((int)rev2.size() - 1);
        std::cout << "Shortest path length = " << hops << "\nPath: ";
        for (int i = (int)rev1.size() - 1; i >= 0; --i)
            std::cout << rev1[i] << ' ';
        for (size_t i = 1; i < rev2.size(); ++i)
            std::cout << rev2[i] << (i+1<rev2.size() ? ' ' : '\n');
    }

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Serial bidirectional BFS took " << elapsed << " seconds\n";

    return 0;
}
