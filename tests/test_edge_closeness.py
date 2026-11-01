"""
Step 0 item 3: closeness centrality checked on small toy graphs against values computed by hand.

Node closeness is nx.closeness_centrality, C(k) = (V - 1) / sum_i d(i, k) on a connected graph (Eq. (8) of the plan),
with distance = 1 / w for weighted graphs and hop counts for binary graphs. Edge closeness of the edge (a, b) is
utils.compute_edge_closeness:

    C_e(a, b) = (E - 1) * C(a) * C(b) / (C(a) + C(b))

the closed form of Broehl & Lehnertz (2022, Sci Rep 12:4407, Eq. 3). That paper's first form,
(E - 1) / sum_i (d(i, a) + d(i, b)), is smaller by the constant factor V - 1, which leaves every weighted Jaccard
similarity unchanged. Compare these values with the code of the Communications Physics paper (its Eq. (9)).

Run from the project root:  .venv/bin/python tests/test_edge_closeness.py   (or with pytest)
"""
import sys
from pathlib import Path

import networkx as nx

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from analysis_pipeline.utils import compute_edge_closeness

TOL = 1e-12


def check(computed: dict, expected: dict, what: str):
    for key, value in expected.items():
        assert abs(computed[key] - value) < TOL, f"{what} {key}: computed {computed[key]!r}, by hand {value!r}"
        print(f"  {what} {key}: computed {computed[key]:.6f} = by hand {value:.6f}")


def test_binary_triangle_with_pendant():
    """
    Binary graph a-b, a-c, b-c (a triangle) plus c-d: V = 4, E = 4.
    Hop distances: a: b=1, c=1, d=2 (sum 4)  b: a=1, c=1, d=2 (sum 4)  c: a=1, b=1, d=1 (sum 3)  d: a=2, b=2, c=1 (sum 5)
    C(a) = 3/4, C(b) = 3/4, C(c) = 3/3 = 1, C(d) = 3/5
    C_e(a, b) = 3 * (3/4 * 3/4) / (3/4 + 3/4) = 9/8
    C_e(a, c) = 3 * (3/4 * 1) / (3/4 + 1) = 9/7,  C_e(b, c) = 9/7
    C_e(c, d) = 3 * (1 * 3/5) / (1 + 3/5) = 9/8
    """
    G = nx.Graph([("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")])
    check(nx.closeness_centrality(G), {"a": 3 / 4, "b": 3 / 4, "c": 1.0, "d": 3 / 5}, "node closeness")
    check(compute_edge_closeness(G, distance=None),
          {("a", "b"): 9 / 8, ("a", "c"): 9 / 7, ("b", "c"): 9 / 7, ("c", "d"): 9 / 8}, "edge closeness")


def test_weighted_triangle_with_indirect_path():
    """
    Weighted triangle with w(x, y) = 0.5, w(y, z) = 0.25, w(x, z) = 1.0, so distances 1/w = 2, 4 and 1: V = 3, E = 3.
    Shortest paths: x-y = 2 (direct; via z 1 + 4 = 5), x-z = 1 (direct), y-z = 3 (via x, 2 + 1; direct 4).
    Distance sums: x: 2 + 1 = 3, y: 2 + 3 = 5, z: 1 + 3 = 4, so C(x) = 2/3, C(y) = 2/5, C(z) = 1/2
    C_e(x, y) = 2 * (2/3 * 2/5) / (2/3 + 2/5) = 1/2
    C_e(x, z) = 2 * (2/3 * 1/2) / (2/3 + 1/2) = 4/7
    C_e(y, z) = 2 * (2/5 * 1/2) / (2/5 + 1/2) = 4/9
    """
    G = nx.Graph()
    G.add_weighted_edges_from([("x", "y", 0.5), ("y", "z", 0.25), ("x", "z", 1.0)])
    for _u, _v, d in G.edges(data=True):
        d["distance"] = 1.0 / d["weight"]  # as in utils.compute_all_broadband_features
    check(nx.closeness_centrality(G, distance="distance"), {"x": 2 / 3, "y": 2 / 5, "z": 1 / 2}, "node closeness")
    check(compute_edge_closeness(G, distance="distance"),
          {("x", "y"): 1 / 2, ("x", "z"): 4 / 7, ("y", "z"): 4 / 9}, "edge closeness")


if __name__ == "__main__":
    for test in (test_binary_triangle_with_pendant, test_weighted_triangle_with_indirect_path):
        print(f"{test.__name__}:")
        test()
    print("All closeness values match the hand computations.")
