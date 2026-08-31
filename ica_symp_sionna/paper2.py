from __future__ import annotations

import csv
import heapq
import json
import math
import shutil
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from common import (
    BUILD, clean_latex_aux, compile_latex, make_release_report,
    occupancy_mask, pdf_pages, write_sha256sums, write_text, zip_directory,
)


PAPER_ID = "ICA_SYMP_2027_Paper_2_Length_Budgeted_MaxMin_RSS_Planning"
TITLE = "Exact Length-Budgeted Max--Min RSS Planning on Sionna Radio Maps"


def neighbors(node: int, ny: int, nx: int):
    y, x = divmod(node, nx)
    if y > 0: yield node - nx
    if y + 1 < ny: yield node + nx
    if x > 0: yield node - 1
    if x + 1 < nx: yield node + 1


def reconstruct(parent: dict[int, int], goal: int) -> list[int]:
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def bfs_path(free: np.ndarray, start: int, goal: int,
             allowed: np.ndarray | None = None) -> list[int] | None:
    ny, nx = free.shape
    if allowed is None:
        allowed = free
    if not allowed.flat[start] or not allowed.flat[goal]:
        return None
    q = deque([start])
    parent: dict[int, int] = {}
    seen = np.zeros(free.size, dtype=bool)
    seen[start] = True
    while q:
        u = q.popleft()
        if u == goal:
            return reconstruct(parent, goal)
        for v in neighbors(u, ny, nx):
            if allowed.flat[v] and not seen[v]:
                seen[v] = True
                parent[v] = u
                q.append(v)
    return None


def weighted_path(free: np.ndarray, rss: np.ndarray, start: int, goal: int,
                  lam: float, reference_dbm: float = -76.0,
                  scale_db: float = 3.0) -> list[int] | None:
    ny, nx = free.shape
    if not free.flat[start] or not free.flat[goal]:
        return None
    dist = np.full(free.size, np.inf)
    dist[start] = 0.0
    parent: dict[int, int] = {}
    heap = [(0.0, start)]
    while heap:
        du, u = heapq.heappop(heap)
        if du != dist[u]:
            continue
        if u == goal:
            return reconstruct(parent, goal)
        for v in neighbors(u, ny, nx):
            if not free.flat[v]:
                continue
            z = (reference_dbm - float(rss.flat[v])) / scale_db
            penalty = math.log1p(math.exp(min(50.0, max(-50.0, z))))
            nd = du + 1.0 + lam * penalty
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return None


def path_metrics(path: list[int], rss: np.ndarray, cell_m: float,
                 outage_threshold_dbm: float = -80.0) -> dict:
    vals = rss.flat[np.asarray(path, dtype=int)]
    longest = current = 0
    for value in vals:
        if value < outage_threshold_dbm:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "steps": len(path)-1,
        "length_m": (len(path)-1) * cell_m,
        "minimum_rss_dbm": float(np.min(vals)),
        "mean_rss_dbm": float(np.mean(vals)),
        "outage_run_m": float(max(0, longest-1) * cell_m),
    }


def exact_budgeted_maxmin(free: np.ndarray, rss: np.ndarray,
                          start: int, goal: int, budget_steps: int) -> tuple[list[int], float]:
    """Exact finite-graph bottleneck path under an edge-count budget.

    For threshold q, a path whose minimum node RSS is at least q exists within
    the budget exactly when the shortest path in the q-induced graph has no
    more than budget_steps edges. Feasibility is monotone in q, permitting a
    binary search over the finite set of node RSS values.
    """
    values = np.unique(rss[free & np.isfinite(rss)])
    if len(values) == 0:
        raise RuntimeError("No finite RSS values on the free graph")
    lo, hi = 0, len(values)-1
    best_path = None
    best_q = float(values[0])
    while lo <= hi:
        mid = (lo + hi) // 2
        q = float(values[mid])
        allowed = free & (rss >= q)
        path = bfs_path(free, start, goal, allowed)
        feasible = path is not None and len(path)-1 <= budget_steps
        if feasible:
            best_path, best_q = path, q
            lo = mid + 1
        else:
            hi = mid - 1
    if best_path is None:
        path = bfs_path(free, start, goal)
        if path is None or len(path)-1 > budget_steps:
            raise RuntimeError("No path satisfies the supplied length budget")
        best_path = path
        best_q = float(np.min(rss.flat[np.asarray(path)]))
    # Recompute the shortest path on the final threshold to bind the returned
    # geometry to the threshold certificate.
    final = bfs_path(free, start, goal, free & (rss >= best_q - 1e-10))
    if final is None or len(final)-1 > budget_steps:
        raise AssertionError("Threshold certificate and returned path disagree")
    return final, float(np.min(rss.flat[np.asarray(final)]))


def snap_node(xgrid: np.ndarray, ygrid: np.ndarray, free: np.ndarray,
              point: tuple[float, float]) -> int:
    xx, yy = np.meshgrid(xgrid, ygrid)
    d2 = (xx-point[0])**2 + (yy-point[1])**2
    d2 = np.where(free, d2, np.inf)
    idx = int(np.argmin(d2))
    if not np.isfinite(d2.flat[idx]):
        raise RuntimeError(f"Cannot snap point {point} to a free node")
    return idx


def exactness_test() -> dict:
    rng = np.random.default_rng(1618033)
    checked = 0
    max_gap = 0.0
    for case in range(36):
        ny, nx = 3, 4
        free = rng.random((ny, nx)) > 0.18
        free[0, 0] = True; free[-1, -1] = True
        s, t = 0, ny*nx-1
        shortest = bfs_path(free, s, t)
        if shortest is None:
            continue
        budget = min(ny*nx-1, len(shortest)-1 + 3)
        rss = rng.normal(-75.0, 7.0, size=(ny, nx))
        alg_path, alg_q = exact_budgeted_maxmin(free, rss, s, t, budget)

        best = -np.inf
        seen = {s}
        def dfs(u: int, path: list[int]):
            nonlocal best
            if len(path)-1 > budget:
                return
            if u == t:
                best = max(best, float(np.min(rss.flat[np.asarray(path)])))
                return
            for v in neighbors(u, ny, nx):
                if free.flat[v] and v not in seen:
                    seen.add(v); path.append(v)
                    dfs(v, path)
                    path.pop(); seen.remove(v)
        dfs(s, [s])
        if not np.isfinite(best):
            raise AssertionError("Exhaustive test failed to find the known shortest path")
        gap = abs(best - alg_q)
        max_gap = max(max_gap, gap)
        if gap > 1e-10 or len(alg_path)-1 > budget:
            raise AssertionError(f"Exactness mismatch in case {case}: {best} versus {alg_q}")
        checked += 1
    return {"cases_checked": checked, "maximum_bottleneck_gap_db": max_gap}


def run_experiment(dataset_npz: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset_npz)
    x, y = data["x"], data["y"]
    rss = gaussian_filter(data["p2_best_server"].astype(float), 0.55)
    occ = occupancy_mask(x, y, clearance=0.24)
    free = (~occ) & np.isfinite(rss) & (rss > -130.0)
    cell = float(np.median(np.diff(x)))

    left_points = [(0.9, 2.0), (0.9, 5.2), (0.9, 9.2), (0.9, 13.0)]
    right_points = [(19.1, 2.0), (19.1, 5.2), (19.1, 9.2), (19.1, 13.0)]
    pairs = [(snap_node(x, y, free, a), snap_node(x, y, free, b), a, b)
             for a in left_points for b in right_points]
    budget_ratio = 1.30
    records = []
    paths: dict[str, list[int]] = {}

    for pair_id, (start, goal, _, _) in enumerate(pairs):
        shortest = bfs_path(free, start, goal)
        if shortest is None:
            raise RuntimeError(f"Boundary pair {pair_id} is disconnected")
        shortest_steps = len(shortest)-1
        budget_steps = int(math.floor(budget_ratio*shortest_steps + 1e-9))
        exact, q = exact_budgeted_maxmin(free, rss, start, goal, budget_steps)
        widest, _ = exact_budgeted_maxmin(free, rss, start, goal, free.size)

        weighted_candidates = []
        for lam in [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.75, 1.25, 2.0, 3.5, 6.0, 10.0]:
            p = weighted_path(free, rss, start, goal, lam)
            if p is not None and len(p)-1 <= budget_steps:
                m = path_metrics(p, rss, cell)
                weighted_candidates.append((m["minimum_rss_dbm"], -m["steps"], lam, p))
        if not weighted_candidates:
            raise RuntimeError(f"No feasible weighted baseline for pair {pair_id}")
        _, _, best_lam, weighted = max(weighted_candidates, key=lambda z: (z[0], z[1]))

        method_paths = {
            "Shortest": shortest,
            "Weighted": weighted,
            "Exact budgeted": exact,
            "Unconstrained widest": widest,
        }
        for method, path in method_paths.items():
            m = path_metrics(path, rss, cell)
            records.append({
                "pair": pair_id, "method": method,
                "steps": m["steps"], "length_m": m["length_m"],
                "length_ratio": m["steps"] / shortest_steps,
                "minimum_rss_dbm": m["minimum_rss_dbm"],
                "mean_rss_dbm": m["mean_rss_dbm"],
                "outage_run_m": m["outage_run_m"],
                "budget_steps": budget_steps,
                "weighted_lambda": best_lam if method == "Weighted" else "",
            })
            paths[f"pair{pair_id}_{method}"] = path
        if len(exact)-1 > budget_steps:
            raise AssertionError("Exact planner violated the length budget")
        if q + 1e-9 < path_metrics(exact, rss, cell)["minimum_rss_dbm"]:
            raise AssertionError("Threshold certificate is below returned bottleneck")

    with (out / "planning_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    summaries = {}
    for method in ["Shortest", "Weighted", "Exact budgeted", "Unconstrained widest"]:
        rr = [r for r in records if r["method"] == method]
        summaries[method] = {
            "median_minimum_rss_dbm": float(np.median([r["minimum_rss_dbm"] for r in rr])),
            "mean_minimum_rss_dbm": float(np.mean([r["minimum_rss_dbm"] for r in rr])),
            "median_length_ratio": float(np.median([r["length_ratio"] for r in rr])),
            "median_outage_run_m": float(np.median([r["outage_run_m"] for r in rr])),
        }

    gains = []
    for pair_id in range(len(pairs)):
        vals = {r["method"]: r for r in records if r["pair"] == pair_id}
        gains.append(vals["Exact budgeted"]["minimum_rss_dbm"] - vals["Shortest"]["minimum_rss_dbm"])
    representative = int(np.argmax(gains))
    start, goal, _, _ = pairs[representative]
    shortest_steps = next(int(r["steps"]) for r in records
                          if r["pair"] == representative and r["method"] == "Shortest")
    frontier = []
    for ratio in np.linspace(1.0, 1.60, 13):
        budget = int(math.floor(ratio*shortest_steps + 1e-9))
        p, q = exact_budgeted_maxmin(free, rss, start, goal, budget)
        frontier.append({"budget_ratio": float(budget/shortest_steps),
                         "minimum_rss_dbm": float(q), "steps": len(p)-1})
    with (out / "budget_frontier.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frontier[0]))
        writer.writeheader(); writer.writerows(frontier)

    exact_test = exactness_test()
    results = {
        "number_pairs": len(pairs),
        "budget_ratio": budget_ratio,
        "cell_size_m": cell,
        "outage_threshold_dbm": -80.0,
        "summary": summaries,
        "representative_pair": representative,
        "representative_gain_over_shortest_db": float(gains[representative]),
        "median_gain_over_shortest_db": float(np.median(gains)),
        "all_exact_paths_within_budget": all(
            int(r["steps"]) <= int(r["budget_steps"])
            for r in records if r["method"] == "Exact budgeted"),
        "exactness_test": exact_test,
    }
    write_text(out / "results.json", json.dumps(results, indent=2) + "\n")

    arrays = {"x": x, "y": y, "rss_dbm": rss, "occupancy": occ,
              "free": free, "representative_pair": representative}
    for key, path in paths.items():
        arrays[key.replace(" ", "_")] = np.asarray(path, dtype=int)
    np.savez_compressed(out / "planning_arrays.npz", **arrays)

    path_rows = []
    ny, nx = free.shape
    for key, path in paths.items():
        pair_name, method = key.split("_", 1)
        for order, node in enumerate(path):
            iy, ix = divmod(node, nx)
            path_rows.append({"pair": int(pair_name[4:]), "method": method,
                              "order": order, "node": node,
                              "x_m": float(x[ix]), "y_m": float(y[iy]),
                              "rss_dbm": float(rss[iy, ix])})
    with (out / "path_coordinates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(path_rows[0]))
        writer.writeheader(); writer.writerows(path_rows)
    return results


def make_figures(results_dir: Path, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    arr = np.load(results_dir / "planning_arrays.npz")
    results = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    x, y, rss, occ = arr["x"], arr["y"], arr["rss_dbm"], arr["occupancy"]
    ny, nx = rss.shape
    extent = [x[0]-0.25, x[-1]+0.25, y[0]-0.25, y[-1]+0.25]
    pair = int(results["representative_pair"])

    fig, ax = plt.subplots(figsize=(7.05, 3.15), constrained_layout=True)
    im = ax.imshow(rss, origin="lower", extent=extent, aspect="equal")
    ax.contour(x, y, occ.astype(float), levels=[0.5], linewidths=1.0)
    markers = {"Shortest": "o", "Weighted": "s", "Exact_budgeted": "^",
               "Unconstrained_widest": "d"}
    for method in ["Shortest", "Weighted", "Exact_budgeted", "Unconstrained_widest"]:
        key = f"pair{pair}_{method}"
        path = arr[key]
        pts = np.asarray([(x[node % nx], y[node // nx]) for node in path])
        label = method.replace("_", " ")
        ax.plot(pts[:, 0], pts[:, 1], marker=markers[method], markevery=max(1, len(path)//10),
                markersize=3, linewidth=1.2, label=label)
    first = arr[f"pair{pair}_Shortest"]
    s, t = int(first[0]), int(first[-1])
    ax.scatter([x[s % nx], x[t % nx]], [y[s // nx], y[t // nx]],
               marker="x", s=48, linewidths=1.5, label="start/goal")
    ax.scatter([2.5, 17.5], [2.5, 12.5], marker="*", s=65, label="APs")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Representative boundary pair {pair}")
    ax.legend(loc="upper center", ncol=3, fontsize=7, frameon=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Best-server RSS (dBm)")
    fig.savefig(figures / "environment_and_paths.pdf", bbox_inches="tight")
    fig.savefig(figures / "environment_and_paths.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    rows = list(csv.DictReader((results_dir / "planning_trials.csv").open(encoding="utf-8")))
    methods = ["Shortest", "Weighted", "Exact budgeted", "Unconstrained widest"]
    short = ["SP", "WP", "EB", "UW"]
    min_rss = [[float(r["minimum_rss_dbm"]) for r in rows if r["method"] == m] for m in methods]
    ratios = [[float(r["length_ratio"]) for r in rows if r["method"] == m] for m in methods]
    frontier = list(csv.DictReader((results_dir / "budget_frontier.csv").open(encoding="utf-8")))

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), constrained_layout=True)
    axes[0].boxplot(min_rss, tick_labels=short, showfliers=False)
    axes[0].set_ylabel("Path minimum RSS (dBm)")
    axes[0].set_title("Sixteen boundary pairs")
    axes[1].boxplot(ratios, tick_labels=short, showfliers=False)
    axes[1].axhline(results["budget_ratio"], linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Length / shortest length")
    axes[1].set_title("Route length")
    axes[2].step([float(r["budget_ratio"]) for r in frontier],
                 [float(r["minimum_rss_dbm"]) for r in frontier], where="post")
    axes[2].set_xlabel("Length budget ratio")
    axes[2].set_ylabel("Optimal minimum RSS (dBm)")
    axes[2].set_title("Exact bottleneck frontier")
    fig.savefig(figures / "planning_performance.pdf", bbox_inches="tight")
    fig.savefig(figures / "planning_performance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def bibliography() -> str:
    return r"""
@misc{nimier2023sionnart,
  author={M. Nimier-David and others},
  title={Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling},
  year={2023}, eprint={2303.11103}, archivePrefix={arXiv}
}
@misc{hoydis2022sionna,
  author={J. Hoydis and others},
  title={Sionna: An Open-Source Library for Next-Generation Physical Layer Research},
  year={2022}, eprint={2203.11854}, archivePrefix={arXiv}
}
@article{ghaffarkhah2011communication,
  author={A. Ghaffarkhah and Y. Mostofi},
  title={Communication-Aware Motion Planning in Mobile Networks},
  journal={IEEE Transactions on Automatic Control}, volume={56}, number={10},
  pages={2478--2485}, year={2011}
}
@article{zavlanos2011graph,
  author={M. M. Zavlanos and M. B. Egerstedt and G. J. Pappas},
  title={Graph-Theoretic Connectivity Control of Mobile Robot Networks},
  journal={Proceedings of the IEEE}, volume={99}, number={9}, pages={1525--1540}, year={2011}
}
@article{dijkstra1959note,
  author={E. W. Dijkstra}, title={A Note on Two Problems in Connexion with Graphs},
  journal={Numerische Mathematik}, volume={1}, pages={269--271}, year={1959},
  doi={10.1007/BF01386390}
}
@article{pollack1960maximum,
  author={M. Pollack}, title={The Maximum Capacity Through a Network},
  journal={Operations Research}, volume={8}, number={5}, pages={733--736}, year={1960}
}
@article{handler1980dual,
  author={G. Y. Handler and I. Zang}, title={A Dual Algorithm for the Constrained Shortest Path Problem},
  journal={Networks}, volume={10}, number={4}, pages={293--309}, year={1980},
  doi={10.1002/net.3230100403}
}
@book{bertsekas1998network,
  author={D. P. Bertsekas}, title={Network Optimization: Continuous and Discrete Models},
  publisher={Athena Scientific}, year={1998}
}
@techreport{3gpp38901,
  author={{3GPP}}, title={Study on Channel Model for Frequencies from 0.5 to 100 GHz},
  institution={3rd Generation Partnership Project}, number={TR 38.901}, year={2024}
}
"""


def render_tex(results: dict, level: int) -> str:
    s = results["summary"]
    sp, wp, eb, uw = s["Shortest"], s["Weighted"], s["Exact budgeted"], s["Unconstrained widest"]
    gain_sp = eb["median_minimum_rss_dbm"] - sp["median_minimum_rss_dbm"]
    gain_wp = eb["median_minimum_rss_dbm"] - wp["median_minimum_rss_dbm"]
    extra_proof = "" if level < 1 else r"""
The proof does not require nonnegative RSS values or a particular propagation
model.  It uses only a fixed scalar value at each graph node and a nonnegative
edge length.  The same construction therefore applies to SINR, achievable
rate, or an outage margin after replacing the node field and retaining the
monotone threshold test.
"""
    extra_discussion = "" if level < 2 else r"""
The planner assumes that the map is known during a mission.  Map uncertainty
could instead be represented by a lower confidence bound at each node, in
which case the same algorithm would optimize a conservative bottleneck.
Time-varying blockage would require replanning or a space--time graph and is
outside the evidence reported here.
"""
    extra_repro = "" if level < 3 else r"""
The distributed implementation archives every path as ordered grid
coordinates, not only the plotted representative route.  It also compares the
threshold algorithm with exhaustive enumeration on randomly obstructed small
graphs, which isolates the discrete optimality claim from Sionna RT and from
the selected indoor geometry.
"""
    tight = r"""
\setlength{\textfloatsep}{6pt plus 1pt minus 2pt}
\setlength{\floatsep}{5pt plus 1pt minus 2pt}
\setlength{\intextsep}{5pt plus 1pt minus 2pt}
\setlength{\abovedisplayskip}{4pt plus 1pt minus 2pt}
\setlength{\belowdisplayskip}{4pt plus 1pt minus 2pt}
""" if level == -1 else ""
    return rf"""
\documentclass[conference]{{IEEEtran}}
\usepackage{{amsmath,amssymb,graphicx,booktabs,microtype,balance}}
\usepackage[hidelinks]{{hyperref}}
{tight}
\graphicspath{{{{../figures/}}}}
\title{{{TITLE}}}
\author{{\IEEEauthorblockN{{Jake W. Liu}}
\IEEEauthorblockA{{Department of Electronic Engineering\\
National Taipei University of Technology, Taipei, Taiwan}}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Radio-aware robot planning is often implemented by adding a tunable received-signal-strength (RSS) penalty to geometric path length.  Such scalarization does not certify the worst communication condition or enforce an interpretable detour limit.  This paper formulates finite-grid planning as maximization of the minimum node RSS subject to an explicit path-length budget.  For each candidate RSS threshold, feasibility is equivalent to a shortest-path query on the induced subgraph; monotonicity permits an exact binary search over the finite RSS values.  A two-access-point Sionna RT 2.0.1 map supplies the site-specific node field.  Across a complete set of 16 boundary start--goal pairs with a 1.30 shortest-length budget, the exact budgeted planner improves median minimum RSS by {gain_sp:.2f}~dB over geometric shortest paths and by {gain_wp:.2f}~dB over a tuned weighted-cost baseline.  Exactness is restricted to the archived occupancy graph, edge lengths, and RSS map.
\end{{abstract}}
\begin{{IEEEkeywords}}
radio-aware navigation, Sionna RT, bottleneck path, constrained planning, wireless digital twin
\end{{IEEEkeywords}}

\section{{Introduction}}
Mobile robots in warehouses, hospitals, and industrial facilities may depend on a wireless link for supervision, map updates, or off-board computation.  A geometric shortest path can cross a radio shadow even when a modest detour maintains connectivity.  Communication-aware motion planning consequently combines mobility and link quality~\cite{{ghaffarkhah2011communication,zavlanos2011graph}}.  Site-specific propagation tools such as Sionna RT make this coupling explicit by placing a radio map on the same geometry used by the planner~\cite{{nimier2023sionnart,hoydis2022sionna}}.

A common implementation minimizes path length plus a weighted RSS or outage penalty.  The weight has no direct operational meaning and different weights may produce the same route or skip a useful route entirely.  A constrained shortest-path formulation provides an explicit resource budget but is generally more involved~\cite{{handler1980dual,bertsekas1998network}}.  Here the communication objective is a bottleneck: maximize the weakest RSS encountered along the path while limiting length.  This structure admits a simpler exact solution than a general additive constrained path.

The contribution is threefold.  First, the route requirement is stated as an interpretable maximum detour relative to the geometric shortest path.  Second, an exact finite-graph algorithm is obtained by thresholding the radio map and applying ordinary shortest-path search.  Third, the method is evaluated on all 16 prespecified left-to-right boundary pairs of a Sionna RT indoor map against geometric, weighted, and unconstrained widest-path baselines.  The word \emph{{exact}} is used only for the discrete graph problem and not for continuous robot dynamics or propagation uncertainty.

\section{{Length-Budgeted Bottleneck Planning}}
Let $G=(V,E)$ be the four-neighbor free-space graph, $\ell(e)>0$ an edge length, and $r(v)$ node RSS in dBm.  For a start $s$, goal $t$, and length budget $B$, the problem is
\begin{{equation}}
 \max_{{P:s\leadsto t}}\; b(P)=\min_{{v\in P}} r(v)
 \quad\text{{s.t.}}\quad L(P)=\sum_{{e\in P}}\ell(e)\le B.
 \label{{eq:problem}}
\end{{equation}}
For threshold $q$, define the induced graph $G_q$ containing nodes with $r(v)\ge q$, and let $d_q(s,t)$ be its shortest-path length, with $d_q=\infty$ when disconnected.  Then
\begin{{equation}}
 q^\star=\max\{{q\in r(V): d_q(s,t)\le B\}}.
 \label{{eq:threshold}}
\end{{equation}}
A path feasible in $G_q$ has bottleneck at least $q$.  Conversely, every path with bottleneck at least $q$ lies entirely in $G_q$.  Equation~\eqref{{eq:threshold}} is therefore equivalent to~\eqref{{eq:problem}}.  Since feasibility is monotone as $q$ decreases, sorting the distinct node values and applying binary search requires $O(\log |V|)$ shortest-path queries.  With a uniform grid, each query is breadth-first search, giving $O((|V|+|E|)\log |V|)$ time after sorting.  The final breadth-first search also returns the shortest route among routes attaining $q^\star$.
{extra_proof}

The budget is $B=1.30L_0$, where $L_0$ is the obstacle-aware geometric shortest length.  Three baselines are used.  The shortest-path baseline minimizes length.  The weighted baseline minimizes the sum of unit edge cost and a softplus penalty on RSS below $-76$~dBm; twelve weights are swept, and the feasible route with the largest bottleneck is retained.  The unconstrained widest path applies~\eqref{{eq:threshold}} without the length restriction and therefore upper-bounds communication quality but can violate the operational detour limit.

\section{{Sionna RT Map and Evaluation}}
The $20\times15\times3$-m digital twin contains concrete walls, a wood partition, and a metal cabinet.  Two isotropic access points at opposite corners transmit 20~dBm at 3.5~GHz.  Sionna RT computes 0.5-m radio-map cells with five interaction depths, line-of-sight, specular reflection, and refraction.  The planner uses best-server RSS, $r(v)=\max_k r_k(v)$, after a fixed 0.55-cell smoothing operation.  Obstacles are dilated by 0.24~m before graph construction to represent robot clearance.

\begin{{figure}}[t]
 \centering
 \includegraphics[width=\columnwidth]{{environment_and_paths.pdf}}
 \caption{{Best-server Sionna RT map and four routes for the representative boundary pair.  The exact budgeted route avoids the principal radio depression without taking the full unconstrained detour.}}
 \label{{fig:p2paths}}
\end{{figure}}

The evaluation uses the Cartesian product of four fixed left-boundary and four fixed right-boundary locations.  Every location is snapped to its nearest free cell, and no pair is removed after observing performance.  Reported metrics are minimum path RSS, route length normalized by $L_0$, and the longest contiguous distance below $-80$~dBm.  All baseline paths and selected weighted-penalty values are archived.  Discrete exactness is additionally checked on randomly obstructed $3\times4$ graphs by exhaustive enumeration of every simple path within the budget.

\begin{{figure}}[t]
 \centering
 \includegraphics[width=\columnwidth]{{planning_performance.pdf}}
 \caption{{Minimum RSS and length distributions for shortest (SP), weighted (WP), exact budgeted (EB), and unconstrained widest (UW) planning, together with the exact budget frontier of the representative pair.}}
 \label{{fig:p2perf}}
\end{{figure}}

\section{{Results and Discussion}}
Table~\ref{{tab:p2results}} shows that exact budgeted planning raises the median minimum RSS from {sp['median_minimum_rss_dbm']:.2f} to {eb['median_minimum_rss_dbm']:.2f}~dBm relative to the shortest route.  The tuned weighted baseline reaches {wp['median_minimum_rss_dbm']:.2f}~dBm.  Every exact budgeted route satisfies the 1.30 limit by construction, with median length ratio {eb['median_length_ratio']:.3f}.  The unconstrained widest path attains {uw['median_minimum_rss_dbm']:.2f}~dBm but has median length ratio {uw['median_length_ratio']:.3f}; it is a communication upper bound rather than a feasible operational baseline.

\begin{{table}}[t]
\caption{{Aggregate performance for 16 start--goal pairs}}
\label{{tab:p2results}}
\centering
\begin{{tabular}}{{lccc}}
\toprule
Method & Median min. & Median length & Median outage \\
 & RSS (dBm) & ratio & run (m) \\
\midrule
Shortest & {sp['median_minimum_rss_dbm']:.2f} & {sp['median_length_ratio']:.3f} & {sp['median_outage_run_m']:.2f} \\
Weighted & {wp['median_minimum_rss_dbm']:.2f} & {wp['median_length_ratio']:.3f} & {wp['median_outage_run_m']:.2f} \\
Exact budgeted & {eb['median_minimum_rss_dbm']:.2f} & {eb['median_length_ratio']:.3f} & {eb['median_outage_run_m']:.2f} \\
Unconstr. widest & {uw['median_minimum_rss_dbm']:.2f} & {uw['median_length_ratio']:.3f} & {uw['median_outage_run_m']:.2f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

The staircase in Fig.~\ref{{fig:p2perf}} follows directly from the finite set of node RSS values.  It exposes the marginal benefit of additional path length without tuning a penalty coefficient.  The weighted sweep is comparatively strong because its coefficient is selected retrospectively from twelve candidates subject to the same budget, yet scalarization is not guaranteed to recover every bottleneck-optimal route.  By contrast,~\eqref{{eq:threshold}} certifies the optimum on the archived graph.  The exhaustive tests report zero bottleneck discrepancy over all checked small-graph cases.

The evidence remains simulation based.  Radio-map error, localization error, fading around cell centers, motion constraints, and dynamic obstacles are absent.  The graph uses four-neighbor translation and a static best-server map; hence exactness does not extend to a continuous curvature-constrained robot or to an uncertain future channel.  The propagation settings also omit diffraction and diffuse scattering, and field validation is required before operational use.
{extra_discussion}
{extra_repro}

\section{{Conclusion}}
A length-budgeted max--min RSS problem on a site-specific occupancy graph was reduced to monotone shortest-path feasibility over radio-map thresholds.  The resulting algorithm is exact for the finite graph and replaces a penalty weight with an explicit detour limit.  Sionna RT experiments show how the method trades route length for the weakest predicted communication condition across a complete boundary-pair set.
\balance
\bibliographystyle{{IEEEtran}}
\bibliography{{refs}}
\end{{document}}
"""


def build_package(dataset_dir: Path, destination: Path) -> Path:
    package = BUILD / PAPER_ID
    if package.exists():
        shutil.rmtree(package)
    (package / "manuscript").mkdir(parents=True)
    (package / "figures").mkdir()
    (package / "data").mkdir()
    (package / "code").mkdir()
    (package / "scene").mkdir()
    (package / "verification").mkdir()

    results = run_experiment(dataset_dir / "sionna_radio_maps.npz", package / "data")
    make_figures(package / "data", package / "figures")
    shutil.copy2(dataset_dir / "sionna_radio_maps.npz", package / "data" / "sionna_radio_maps.npz")
    shutil.copy2(dataset_dir / "sionna_metadata.json", package / "data" / "sionna_metadata.json")
    shutil.copytree(dataset_dir / "scene", package / "scene", dirs_exist_ok=True)
    shutil.copy2(Path(__file__), package / "code" / "paper2.py")
    shutil.copy2(Path(__file__).with_name("common.py"), package / "code" / "common.py")
    write_text(package / "code" / "run_all.py", """from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nfrom paper2 import run_experiment, make_figures\nroot=Path(__file__).resolve().parents[1]\nrun_experiment(root/'data'/'sionna_radio_maps.npz', root/'data')\nmake_figures(root/'data', root/'figures')\nprint('Paper 2 numerical results and figures regenerated.')\n""")
    write_text(package / "requirements.txt", "numpy\nscipy\nmatplotlib\npypdf\nsionna-rt==2.0.1\n")
    write_text(package / "manuscript" / "refs.bib", bibliography())

    chosen = None
    for level in [-1, 0, 1, 2, 3]:
        write_text(package / "manuscript" / "main.tex", render_tex(results, level))
        pdf = compile_latex(package / "manuscript")
        if pdf_pages(pdf) == 4:
            chosen = level
            break
    if chosen is None:
        pages = pdf_pages(package / "manuscript" / "main.pdf")
        raise AssertionError(f"Paper 2 did not compile to four pages; obtained {pages}")
    shutil.copy2(package / "manuscript" / "main.pdf", package / "paper.pdf")
    clean_latex_aux(package / "manuscript")

    et = results["exactness_test"]
    tests = {
        "pdf_exactly_four_pages": pdf_pages(package / "paper.pdf") == 4,
        "all_reported_paths_within_budget": results["all_exact_paths_within_budget"],
        "exhaustive_small_graph_cases_positive": et["cases_checked"] >= 20,
        "exhaustive_maximum_gap_below_1e-10": et["maximum_bottleneck_gap_db"] < 1e-10,
        "all_sixteen_pairs_present": results["number_pairs"] == 16,
    }
    if not all(tests.values()):
        raise AssertionError(f"Paper 2 verification failed: {tests}")

    write_text(package / "README.md", f"""# {TITLE}

This directory is the complete reproducibility package for a four-page IEEE conference manuscript.

## Reproduce the numerical results

```bash
python -m pip install -r requirements.txt
python code/run_all.py
```

The archived `data/sionna_radio_maps.npz` contains the two access-point maps generated with Sionna RT 2.0.1.  The Python pipeline rebuilds the occupancy graph, all 16 start--goal experiments, every route coordinate, the exactness tests, the result tables, and both figures.

## Compile the manuscript

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The distributed `paper.pdf` contains exactly four pages including references.
""")
    write_text(package / "verification" / "claim_evidence_ledger.md", """# Claim--evidence ledger

| Manuscript claim | Direct evidence |
|---|---|
| Exact threshold characterization on the finite graph | `code/paper2.py::exact_budgeted_maxmin` and proof in `manuscript/main.tex` |
| Exhaustive exactness check | `data/results.json` and `code/paper2.py::exactness_test` |
| Two-AP best-server Sionna RT field | `data/sionna_radio_maps.npz`, `data/sionna_metadata.json`, and `scene/` |
| Sixteen prespecified boundary pairs | Trial-level `data/planning_trials.csv` |
| Path geometry and budget compliance | `data/path_coordinates.csv` and `data/planning_arrays.npz` |
| Figures and aggregate values | Generated from the same CSV/NPZ/JSON data by `code/run_all.py` |
| Four-page length | `verification/release_check.json` and `paper.pdf` |
""")
    write_text(package / "verification" / "novelty_boundary.md", """# Novelty boundary

The paper does not claim that communication-aware planning, widest paths, shortest paths, constrained paths, or Sionna RT radio maps are individually new.  The scoped contribution is an explicit length-budgeted max--min RSS formulation, its exact finite-graph threshold solution, and its evaluation on a site-specific Sionna RT best-server map against an adaptively tuned scalarized baseline.

Targeted search concepts used in preparing the manuscript were: `communication-aware robot path planning RSS`, `wireless connectivity constrained shortest path`, `widest path length constraint`, `bottleneck path detour budget`, `Sionna RT robot navigation`, and `radio map path planning`.  The paper avoids a first-ever claim and restricts exactness to the stated graph.  This boundary assessment is not a proof that no unpublished or differently worded work exists.
""")
    write_text(package / "verification" / "reference_check.md", """# Reference check

The bibliography separates the Sionna/Sionna RT propagation platform, communication-aware robotics, graph connectivity control, shortest/widest path foundations, and constrained-path literature.  The discrete optimality claim is proved in the manuscript and tested in code rather than delegated to a citation.  DOI-bearing entries retain their DOI in `manuscript/refs.bib`.
""")
    make_release_report(package, TITLE, "Sionna RT 2.0.1 plus exact finite-grid graph planning", tests)
    write_sha256sums(package)
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / f"{PAPER_ID}_full_source.zip"
    zip_directory(package, zip_path)
    return zip_path
