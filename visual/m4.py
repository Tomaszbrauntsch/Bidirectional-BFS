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

    # ——— 1) load your path from JSON, extract plain ints —————
    with open('../graphs/1k.json') as f:
        raw = json.load(f)
        # assume raw['nodes'] is a list of dicts like {'id': 5, …}
        path_nodes = [int(nd['id']) for nd in raw['nodes']]

    # build the list of (u,v) pairs
    marked_path = list(zip(path_nodes, path_nodes[1:]))

    # ——— 2) compute “other” edges if you still want them hidden ———
    # (we won’t draw them, but you could draw them with alpha=0.0)
    other_edges = [
        e for e in G.edges()
        if e not in marked_path and (e[::-1] not in marked_path)
    ]

    # ——— 3) plot ——————————————————————————
    plt.figure(figsize=(8,8))
    pos = nx.kamada_kawai_layout(G)

    # draw all nodes
    nx.draw_networkx_nodes(G, pos, node_size=50)

    # (optional) draw the rest of the edges invisibly so they don’t show:
    nx.draw_networkx_edges(
        G, pos,
        edgelist=other_edges,
        alpha=0.0,
        width=1
    )

    # draw just the path edges in red
    nx.draw_networkx_edges(
        G, pos,
        edgelist=marked_path,
        edge_color='red',
        width=2
    )

    # labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title(f"Graph with {G.number_of_nodes()} nodes, path of length {len(marked_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
