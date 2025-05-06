import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt
import json

def read_graph_csr(path):
    """
    Reads a binary edge‐list [uint32 N, uint32 M, then 2*M uint32s of (u,v) pairs]
    and returns a scipy CSR adjacency matrix for an undirected graph.
    """
    with open(path, 'rb') as f:
        N, M = np.fromfile(f, dtype=np.uint32, count=2)
        edges = np.fromfile(f, dtype=np.uint32, count=2*M).reshape(-1, 2)

    # mirror edges for undirectedness
    all_edges = np.vstack([edges, edges[:, ::-1]])
    row, col = all_edges[:,0], all_edges[:,1]
    data = np.ones_like(row, dtype=np.uint8)
    return csr_matrix((data, (row, col)), shape=(int(N), int(N)))

if __name__ == "__main__":
    # 1) Load CSR
    A = read_graph_csr('../graphs/1k.bin')
    print(f"Loaded CSR with shape {A.shape} and {A.nnz} nonzeros")

    # 2) Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    # add edges directly from the sparse matrix
    G.add_edges_from(zip(*A.nonzero()))

    data = ""
    with open('../graphs/1k.json') as f:
        data = json.load(f)
        data = data['nodes']
        print(data)
    i = 0
    marked_path = list(zip(data, data[1:]))
    other_edges = [e for e in G.edges() if e not in marked_path and (e[::-1] not in marked_path)]
    # print("[marked_path]", marked_path)
    # edge_coloring = []
    # for u, v in G.edges():
    #     if (u, v) in marked_path or (v, u) in marked_path:
    #         edge_coloring.append('red')
    #     else:
    #         edge_coloring.append('black')
    font_size = 10
    # 3) Plot, with each node labeled by its integer ID
    plt.figure(figsize=(8, 8))
    pos = nx.kamada_kawai_layout(G)  # or kamada_kawai_layout, spring_layout, circular_layout, etc.
    nx.draw_networkx_nodes(G, pos, node_size=50)
    
    print("[O_E]", other_edges)
    nx.draw_networkx_edges(G, pos, alpha=0.0, edgelist=other_edges)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edgelist=marked_path, edge_color='red', width=5.0)
    nx.draw_networkx_labels(G, pos, font_size=font_size)  # label each node by its ID

    plt.title(f"Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
