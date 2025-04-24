#!/usr/bin/env python3
import argparse
import json
import os
import struct
import pathlib

import numpy as np
import networkx as nx

# ─── parse args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate a random G(n,p) graph, write its dense adj‑matrix to 50k.bin, "
                "and output the shortest path between src and dst as JSON."
)
parser.add_argument("--n",   type=int, default=50000, help="number of nodes")
parser.add_argument("--p",   type=float, default=3/133333, help="edge probability")
parser.add_argument("--src", type=int, default=0,      help="source node")
parser.add_argument("--dst", type=int, default=None,   help="destination node")
parser.add_argument("--out", type=str, default="50k.bin", help="file to write as")
args = parser.parse_args()

N = args.n
P = args.p
SRC = args.src
DST = args.dst if args.dst is not None else N - 1
PATH = args.out

# ─── generate graph + adjacency ──────────────────────────────────────────────
print(f"Generating G({N}, p={P})…")
G = nx.fast_gnp_random_graph(N, P, directed=False)
A = nx.to_numpy_array(G, dtype=np.uint8, order="C")  # dense 0/1

# ─── write binary matrix ─────────────────────────────────────────────────────
with open(PATH,"wb") as f:
    f.write(struct.pack("<I", N))
    f.write(struct.pack("<I", G.number_of_edges()))
    for u, v in G.edges():
        f.write(struct.pack("<II", u, v))

# ─── compute shortest path ───────────────────────────────────────────────────
print(f"Computing shortest path from {SRC} to {DST}…")
try:
    path_nodes = nx.shortest_path(G, source=SRC, target=DST)
    hop_count = len(path_nodes) - 1
    print(f"  → hop count = {hop_count}")
except nx.NetworkXNoPath:
    path_nodes = []
    hop_count = None
    print("  → no path found")

# ─── dump path JSON ──────────────────────────────────────────────────────────
write_json=PATH.split(".")
out_json = pathlib.Path(write_json[0]+".json")
path_data = {
    "source":    SRC,
    "target":    DST,
    "hop_count": hop_count,
    "nodes":     path_nodes,
}
with out_json.open("w") as f:
    json.dump(path_data, f, indent=2)
print("Wrote path info to", out_json)
