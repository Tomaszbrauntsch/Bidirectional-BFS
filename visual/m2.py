import struct
import networkx as nx
import matplotlib.pyplot as plt
import os

def load_csr(filename):
    """
    Load a graph in CSR format from a binary file.
    Expected layout (little‐endian uint32_t):
      [ n       ]  # number of vertices
      [ nnz     ]  # number of nonzeros (edges)
      [ row_ptr ]  # array of length n+1
      [ col_idx ]  # array of length nnz
    Returns:
      n (int), nnz (int), row_ptr (list of int), col_idx (list of int)
    """
    with open(filename, "rb") as f:
            # 1) Read header
            hdr = f.read(8)
            if len(hdr) < 8:
                raise ValueError("File too short for header")
            n, nnz = struct.unpack("<II", hdr)

            # 2) Read row_ptr
            print("[n]:", n)
            expected_rpb = 4 * (n + 1)
            rpb = f.read(expected_rpb)
            if len(rpb) < expected_rpb:
                raise ValueError(f"File too short for row_ptr: "
                                f"got {len(rpb)}, need {expected_rpb}")
            row_ptr = list(struct.unpack(f"<{n+1}I", rpb))

            # 3) Read col_idx
            expected_cib = 4 * nnz
            cib = f.read(expected_cib)
            if len(cib) < expected_cib:
                raise ValueError(f"File too short for col_idx: "
                                f"got {len(cib)}, need {expected_cib}")
            col_idx = list(struct.unpack(f"<{nnz}I", cib))

        # Optionally check that no extra garbage remains
    actual_size = os.path.getsize(filename)
    expected_size = 8 + expected_rpb + expected_cib
    if actual_size != expected_size:
        print(f"Warning: file size {actual_size} ≠ expected {expected_size}")

    return n, nnz, row_ptr, col_idx

def csr_to_networkx(n, row_ptr, col_idx):
    """
    Convert CSR representation into an undirected networkx Graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for idx in range(row_ptr[u], row_ptr[u+1]):
            v = col_idx[idx]
            G.add_edge(u, v)
    return G

if __name__ == "__main__":
    # 1) Load CSR
    n, nnz, row_ptr, col_idx = load_csr("50k.bin")

    # 2) Build graph
    G = csr_to_networkx(n, row_ptr, col_idx)

    # 3) Visualize
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)          # force‐directed layout
    nx.draw_networkx_nodes(G, pos, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.title(f"Graph with {n} nodes and {nnz} edges")
    plt.axis("off")
    plt.show()
