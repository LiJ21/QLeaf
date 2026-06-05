"""Train an XGBoost model and emit artifacts the qleaf worker benchmark can read.

Produces three files (see `bench/worker.cc`):
  * <out>.xgb.json    raw XGBoost model (the original artifact, kept for reference)
  * <out>.qleaf.json  the forest in qleaf's perfect-tree format  (--trees)
  * <out>.feats.csv   feature rows, one sample per line, CSV       (--features-csv)

qleaf only handles *perfect* trees of a fixed depth, so each (generally pruned)
XGBoost tree is embedded into a complete depth-D tree laid out in level order:
node `i`'s children live at `2i+1` / `2i+2`, internal nodes hold
(split_index, split_threshold) and leaves hold the output value. An XGBoost leaf
that stops short of depth D is expanded into a constant subtree so every root->leaf
path yields that leaf's value. The branch worker decides `feature < split` -> left,
which is exactly XGBoost's "yes" (left_children) direction, so no inversion is needed.
"""

import argparse
import io
import json
import math
import os
import urllib.request

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix, XGBClassifier

# --- known datasets, keyed by the friendly name passed to --dataset ---------
# Each entry maps to an OpenML reference. A --dataset value that isn't a key here
# is forwarded to OpenML as-is: either a dataset name, or "id:<n>" to fetch by
# numeric data_id (the most robust form). The default output basename is derived
# from the dataset and the actual depth/estimators (e.g. "higgs_d6_n500").
#
# "max_rows" caps how many rows to pull *and* triggers the streaming loader
# (stream_arff) instead of fetch_openml. Epsilon's full ARFF is ~10 GB, so we
# stream and stop early by default; --max-rows overrides, --max-rows 0 disables.
DATASETS = {
    "higgs":   {"openml": "Higgs"},                          # 28 numeric features, ~1M rows
    "epsilon": {"openml": "id:45575", "max_rows": 20000},    # 2000 numeric features, 500k rows
}

# --- defaults: the Higgs d6/n500 model from the original script -------------
DEFAULT_DATASET = "higgs"
MAX_DEPTH = 6
N_ESTIMATORS = 500

# Feature rows to dump for the benchmark.
#
# worker_bench reads up to --samples rows (default 1024) and cycles through them
# with `row++ % size`. Using *real* feature values (rather than the uniform-random
# fallback in worker.cc) is the whole point of supplying a CSV: realistic split
# decisions give the BranchRegressionWorker a representative branch-misprediction
# rate. A single repeated row would be perfectly predicted and benchmark
# unrealistically fast.
#
# 1024 matches the benchmark's default --samples and is plenty of path diversity.
# Each Higgs row is 28 float32 = 112 B, so 1024 rows ~= 112 KB: it stays small
# next to the node buffer that actually dominates cache (d6/n500 ~= 500*127*12 B
# ~= 760 KB, about L2-sized), so the feature data adds no cache noise of its own.
N_ROWS = 1024


def base_margin(model: dict) -> float:
    """Margin-space offset XGBoost adds before summing tree outputs."""
    lmp = model["learner"]["learner_model_param"]
    base_score = float(lmp["base_score"].strip("[]"))
    objective = model["learner"]["objective"]["name"]
    if objective == "binary:logistic":
        base_score = min(max(base_score, 1e-7), 1.0 - 1e-7)
        return math.log(base_score / (1.0 - base_score))  # logit link
    # reg:* objectives store the offset directly in margin space.
    return base_score


def convert_tree(tree: dict, depth: int) -> dict:
    """Embed one XGBoost tree into a perfect depth-`depth` tree (level order)."""
    size = (1 << (depth + 1)) - 1
    indices = [0] * size
    splits = [0.0] * size
    left, right = tree["left_children"], tree["right_children"]
    split_idx, split_cond = tree["split_indices"], tree["split_conditions"]

    def fill_constant(pos: int, d: int, value: float) -> None:
        """Pad a short subtree so every leaf under `pos` returns `value`."""
        if d == depth:
            splits[pos] = value
            return
        fill_constant(2 * pos + 1, d + 1, value)
        fill_constant(2 * pos + 2, d + 1, value)

    def fill(pos: int, d: int, node: int) -> None:
        if left[node] == -1:  # XGBoost leaf: value lives in split_conditions
            fill_constant(pos, d, float(split_cond[node]))
            return
        if d == depth:
            raise ValueError("XGBoost tree is deeper than the declared depth")
        indices[pos] = int(split_idx[node])
        splits[pos] = float(split_cond[node])
        fill(2 * pos + 1, d + 1, left[node])   # feature < threshold -> left/yes
        fill(2 * pos + 2, d + 1, right[node])  # else -> right/no

    fill(0, 0, 0)
    return {"indices": indices, "splits": splits}


def to_qleaf_forest(model: dict, depth: int) -> dict:
    """Convert a parsed XGBoost JSON model into a qleaf forest dict."""
    xgb_trees = model["learner"]["gradient_booster"]["model"]["trees"]
    trees = [convert_tree(t, depth) for t in xgb_trees]

    # Fold the base margin into tree 0's leaves so summing all leaves across
    # trees reproduces XGBoost's full output margin (qleaf has no bias term).
    bias = base_margin(model)
    if trees:
        size = (1 << (depth + 1)) - 1
        leaf_start = size - (1 << depth)
        for i in range(leaf_start, size):
            trees[0]["splits"][i] += bias

    return {"depth": depth, "trees": trees}


def qleaf_predict(forest: dict, row) -> float:
    """Reference traversal mirroring BranchRegressionWorker (eps = 0)."""
    depth = forest["depth"]
    total = 0.0
    for tree in forest["trees"]:
        offset = 0
        for _ in range(depth):
            to_left = 1 if row[tree["indices"][offset]] < tree["splits"][offset] else 0
            offset = 2 * (offset + 1) - to_left
        total += tree["splits"][offset]
    return total


def verify(forest: dict, booster, X, n: int = 256) -> None:
    """Assert the converted forest matches XGBoost margins on a sample."""
    n = min(n, len(X))
    expected = booster.predict(DMatrix(X[:n]), output_margin=True)
    got = np.array([qleaf_predict(forest, X[i]) for i in range(n)])
    max_err = float(np.abs(expected - got).max())
    if max_err > 1e-4:
        raise AssertionError(f"forest disagrees with XGBoost margins: {max_err:.3g}")
    print(f"verified: max margin error over {n} rows = {max_err:.3g}")


def resolve_data_id(ref: str) -> int:
    """Map a --dataset reference ('id:<n>', a bare id, or a name) to a data_id."""
    if ref.startswith("id:"):
        return int(ref[3:])
    if ref.isdigit():
        return int(ref)
    url = f"https://www.openml.org/api/v1/json/data/list/data_name/{ref}"
    with urllib.request.urlopen(url, timeout=30) as r:
        listing = json.load(r)
    return int(listing["data"]["dataset"][0]["did"])


def stream_arff(data_id: int, max_rows: int):
    """Download only the first `max_rows` rows of a dense OpenML ARFF.

    fetch_openml always pulls the whole file; Epsilon's is ~10 GB. Here we stream
    the ARFF and stop after `max_rows` data rows, so we transfer only a small
    fraction (Epsilon is ~21 KB/row, served at only ~7 MB/s). Assumes numeric
    feature columns plus a single nominal target column (the one declared with
    `{...}`, located anywhere); returns (X float32, y string array).

    `timeout` is per socket read, so a stalled connection raises instead of
    hanging forever; progress is printed since the transfer can take a while.
    """
    # The download lives at a file_id, not the data_id; grab it from the metadata.
    meta_url = f"https://www.openml.org/api/v1/json/data/{data_id}"
    with urllib.request.urlopen(meta_url, timeout=30) as r:
        file_id = int(json.load(r)["data_set_description"]["file_id"])
    url = f"https://www.openml.org/data/v1/download/{file_id}"
    print(f"streaming first {max_rows} rows of OpenML data_id={data_id} "
          f"(file_id={file_id}, ~7 MB/s) ...", flush=True)

    label_col, n_attr = None, 0
    rows, labels = [], []
    with urllib.request.urlopen(url, timeout=60) as resp:  # closing early aborts xfer
        in_data = False
        for raw in io.TextIOWrapper(resp, encoding="utf-8"):
            if not in_data:
                low = raw.lstrip().lower()
                if low.startswith("@attribute"):
                    if "{" in raw:            # nominal attr -> the target column
                        label_col = n_attr
                    n_attr += 1
                elif low.startswith("@data"):
                    in_data = True
                    if label_col is None:
                        label_col = n_attr - 1  # fall back to the last column
                continue
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split(",")
            labels.append(parts[label_col])
            rows.append([p for i, p in enumerate(parts) if i != label_col])
            if len(rows) % 2000 == 0:
                print(f"  ... {len(rows)}/{max_rows} rows", flush=True)
            if len(rows) >= max_rows:
                break
    return np.asarray(rows, dtype=np.float32), np.asarray(labels)


CACHE_DIR = "/tmp"


def load_subset(data_id: int, max_rows: int):
    """stream_arff backed by an on-disk cache under /tmp, keyed by data_id.

    The OpenML transfer is the slow part (~7 MB/s), so streamed rows are saved
    and reused across runs. A cache holding at least `max_rows` rows is sliced to
    size; a smaller one is re-downloaded and overwritten with the larger set.
    """
    cache = os.path.join(CACHE_DIR, f"qleaf_openml_{data_id}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=False)
        if len(d["X"]) >= max_rows:
            print(f"using cached {cache} ({len(d['X'])} rows, need {max_rows})", flush=True)
            return d["X"][:max_rows], d["y"][:max_rows]
        print(f"cache {cache} has only {len(d['X'])} rows; re-downloading {max_rows}",
              flush=True)
    X, y = stream_arff(data_id, max_rows)
    np.savez(cache, X=X, y=y)
    print(f"cached {len(X)} rows to {cache}", flush=True)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help=f"known dataset ({', '.join(DATASETS)}), or any OpenML "
                         f"name / 'id:<n>'")
    ap.add_argument("--out", default=None,
                    help="output basename (default: per-dataset)")
    ap.add_argument("--depth", type=int, default=MAX_DEPTH)
    ap.add_argument("--estimators", type=int, default=N_ESTIMATORS)
    ap.add_argument("--rows", type=int, default=N_ROWS)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows loaded (streams a subset instead of the full "
                         "download); 0 forces the full dataset. Default: per-dataset")
    ap.add_argument("--parser", default="liac-arff", help="OpenML arff parser")
    args = ap.parse_args()

    entry = DATASETS.get(args.dataset)
    ref = entry["openml"] if entry else args.dataset
    stem = args.dataset.replace("id:", "id").replace("/", "_")
    out = args.out or f"{stem}_d{args.depth}_n{args.estimators}"

    max_rows = args.max_rows if args.max_rows is not None else (entry or {}).get("max_rows")
    if max_rows:  # stream just the first `max_rows` rows (e.g. the 10 GB Epsilon ARFF)
        X, y = load_subset(resolve_data_id(ref), max_rows)
    else:
        print(f"fetching all of {ref} (full download) ...", flush=True)
        kw = {"data_id": resolve_data_id(ref)} if ref.startswith("id:") else {"name": ref}
        X, y = fetch_openml(as_frame=False, return_X_y=True, parser=args.parser, **kw)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = LabelEncoder().fit_transform(y)  # map labels to 0..n-1 (Epsilon ships -1/+1)
    n_features = X.shape[1]
    print(f"{X.shape[0]} rows x {n_features} features", flush=True)

    print(f"training XGBoost depth={args.depth} n_estimators={args.estimators} "
          f"on {X.shape[0]}x{n_features} (this can take a few minutes) ...", flush=True)
    clf = XGBClassifier(
        max_depth=args.depth, n_estimators=args.estimators,
        tree_method="hist", objective="binary:logistic",
        learning_rate=0.1, subsample=0.8,
    ).fit(X, y)
    booster = clf.get_booster()

    xgb_path = f"{out}.xgb.json"
    forest_path = f"{out}.qleaf.json"
    csv_path = f"{out}.feats.csv"

    booster.save_model(xgb_path)
    model = json.loads(booster.save_raw("json"))
    forest = to_qleaf_forest(model, args.depth)
    verify(forest, booster, X)

    with open(forest_path, "w") as f:
        json.dump(forest, f)
    np.savetxt(csv_path, X[: args.rows], delimiter=",", fmt="%.7g")

    print(f"wrote {xgb_path}")
    print(f"wrote {forest_path} ({len(forest['trees'])} trees, depth {args.depth})")
    print(f"wrote {csv_path} ({min(args.rows, len(X))} rows, {n_features} features)")
    print("\nrun the benchmark with:")
    print(
        f"  ./build/worker_bench --trees {os.path.abspath(forest_path)} "
        f"--features {n_features} --features-csv {os.path.abspath(csv_path)} "
        f"--samples {min(args.rows, len(X))} --threads 1,2,4"
    )


if __name__ == "__main__":
    main()
