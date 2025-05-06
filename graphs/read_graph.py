import numpy as np
from scipy.sparse import csr_matrix

def read_graph_csr(path):
    # open in binary mode
    with open(path, 'rb') as f:
        # read N and number of edges M
        N, M = np.fromfile(f, dtype=np.uint32, count=2)

        # read the 2*M uint32’s for the edge list
        edges = np.fromfile(f, dtype=np.uint32, count=2*M)
    # reshape into (M,2)
    edges = edges.reshape(-1, 2)

    # for undirected, mirror (u,v) → (v,u)
    all_edges = np.vstack([edges, edges[:, ::-1]])

    row = all_edges[:, 0]
    col = all_edges[:, 1]
    data = np.ones_like(row, dtype=np.uint8)

    # build and return CSR
    return csr_matrix((data, (row, col)), shape=(int(N), int(N)))

# usage
A = read_graph_csr('1k.bin')
print("Loaded CSR with shape", A.shape, "and", A.nnz, "nonzeros")
rows, cols = A.nonzero()
for u, v in zip(rows, cols):
    print(f"{u} -> {v}")