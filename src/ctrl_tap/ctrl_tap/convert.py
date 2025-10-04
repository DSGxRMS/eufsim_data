#!/usr/bin/env python3
"""
Build centerline midpoints from cone set:
- Left = yellow, Right = blue, ignore orange/big_orange and car_start rows
- Pair L↔R with mutual-NN preference, fallback Hungarian (gated by plausible width)
- Output ordered centerline starting near (0,0)

Usage:
  python build_centerline.py input.csv output.csv
"""

import sys, math, csv
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Please pip install pandas", file=sys.stderr); sys.exit(1)

# Optional (better matching). If unavailable, we fall back to greedy.
try:
    from scipy.spatial import cKDTree as KDTree
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from scipy.optimize import linear_sum_assignment
    HUNGARIAN_OK = True
except ImportError:
    HUNGARIAN_OK = False


# ---- Tunables (keep realistic) ----
MIN_TRACK_WIDTH = 1.5   # meters (hard lower gate)
MAX_TRACK_WIDTH = 8.0   # meters (hard upper gate)
START_XY = np.array([0.0, 0.0])  # car_start
ORDER_MAX_STEP = 5.0     # max hop between sequential center points during ordering (meters). Increase if gaps.

def load_cones(path: Path):
    df = pd.read_csv(path)
    # Normalize tag names (safety)
    df['tag'] = df['tag'].astype(str)
    # Filter
    mask_keep = df['tag'].isin(['yellow', 'blue'])
    df_keep = df.loc[mask_keep, ['tag','x','y']].copy()
    left  = df_keep[df_keep['tag']=='yellow'][['x','y']].to_numpy()
    right = df_keep[df_keep['tag']=='blue'][['x','y']].to_numpy()
    # Try to fetch car_start if present
    car_start = START_XY
    try:
        row = df[(df['tag']=='car_start')].iloc[0]
        car_start = np.array([float(row['x']), float(row['y'])], dtype=float)
    except Exception:
        pass
    return left, right, car_start

def pair_mutual_nn(left: np.ndarray, right: np.ndarray) -> List[Tuple[int,int]]:
    """Fast mutual-NN pairs within width gates."""
    if len(left)==0 or len(right)==0:
        return []

    if SCIPY_OK:
        kdl = KDTree(left); kdr = KDTree(right)
        # Nearest from left->right
        d_lr, j_lr = kdr.query(left, k=1)
        # Nearest from right->left
        d_rl, i_rl = kdl.query(right, k=1)

        pairs = []
        for iL, (jR, d) in enumerate(zip(j_lr, d_lr)):
            if i_rl[jR] == iL and MIN_TRACK_WIDTH <= d <= MAX_TRACK_WIDTH:
                pairs.append((iL, jR))
        return pairs
    else:
        # Brute force mutual-NN
        # right nearest for each left
        dmat = np.linalg.norm(left[:,None,:] - right[None,:,:], axis=2)
        j_lr = np.argmin(dmat, axis=1)
        d_lr = dmat[np.arange(len(left)), j_lr]
        i_rl = np.argmin(dmat, axis=0)
        pairs = []
        for iL, jR in enumerate(j_lr):
            d = d_lr[iL]
            if i_rl[jR] == iL and MIN_TRACK_WIDTH <= d <= MAX_TRACK_WIDTH:
                pairs.append((iL, jR))
        return pairs

def pair_hungarian(left: np.ndarray, right: np.ndarray, existing: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Fill remaining matches using Hungarian within width gate; respects one-to-one by masking used indices."""
    if not HUNGARIAN_OK or len(left)==0 or len(right)==0:
        return existing

    usedL = set(i for i,_ in existing)
    usedR = set(j for _,j in existing)

    Lfree = [i for i in range(len(left)) if i not in usedL]
    Rfree = [j for j in range(len(right)) if j not in usedR]
    if not Lfree or not Rfree:
        return existing

    # Cost matrix with gate
    Lcoords = left[Lfree]
    Rcoords = right[Rfree]
    dmat = np.linalg.norm(Lcoords[:,None,:]-Rcoords[None,:,:], axis=2)
    # Gate with large penalty outside [MIN, MAX]
    penalty = 1e6
    gated = dmat.copy()
    gated[(gated < MIN_TRACK_WIDTH) | (gated > MAX_TRACK_WIDTH)] = penalty

    row_ind, col_ind = linear_sum_assignment(gated)
    for r, c in zip(row_ind, col_ind):
        if gated[r, c] >= penalty:
            continue
        iL = Lfree[r]; jR = Rfree[c]
        existing.append((iL, jR))
    return existing

def compute_midpoints(left: np.ndarray, right: np.ndarray, pairs: List[Tuple[int,int]]) -> np.ndarray:
    mids = []
    for iL, jR in pairs:
        pL = left[iL]; pR = right[jR]
        mids.append((pL + pR)/2.0)
    if not mids:
        return np.zeros((0,2), dtype=float)
    return np.vstack(mids)

def order_path(points: np.ndarray, start_xy: np.ndarray) -> np.ndarray:
    """
    Greedy nearest-neighbor path starting at the point closest to start_xy.
    Works well for skidpad/track centerlines. Breaks ties by distance.
    """
    if len(points)==0:
        return points

    remaining = points.copy()
    # start from nearest to start_xy
    d0 = np.linalg.norm(remaining - start_xy[None,:], axis=1)
    idx0 = int(np.argmin(d0))
    ordered = [remaining[idx0]]
    remaining = np.delete(remaining, idx0, axis=0)

    while len(remaining) > 0:
        last = ordered[-1]
        d = np.linalg.norm(remaining - last[None,:], axis=1)
        j = int(np.argmin(d))
        if d[j] > ORDER_MAX_STEP and len(ordered) > 5:
            # If there's a suspicious gap, we start a new chain only if needed;
            # For standard cone sets this rarely triggers. If it does, relax ORDER_MAX_STEP.
            # Here we still proceed (best effort).
            pass
        ordered.append(remaining[j])
        remaining = np.delete(remaining, j, axis=0)

    return np.vstack(ordered)

def write_csv(points: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['center_x','center_y'])
        for x,y in points:
            w.writerow([f"{x:.6f}", f"{y:.6f}"])

def main():
    if len(sys.argv) < 3:
        print("Usage: python build_centerline.py input.csv output.csv", file=sys.stderr)
        sys.exit(2)

    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])

    left, right, car_start = load_cones(inp)
    if len(left)==0 or len(right)==0:
        print("ERROR: Need both yellow (left) and blue (right) cones.", file=sys.stderr)
        sys.exit(3)

    # Step 1: mutual nearest neighbors within gates
    pairs = pair_mutual_nn(left, right)

    # Step 2: fill gaps via Hungarian (within gates)
    pairs = pair_hungarian(left, right, pairs)

    if not pairs:
        print("ERROR: No valid L-R pairs found within width gates.", file=sys.stderr)
        sys.exit(4)

    # Step 3: midpoints
    mids = compute_midpoints(left, right, pairs)

    # Step 4: order path from car_start
    mids_ord = order_path(mids, car_start)

    # Output
    write_csv(mids_ord, outp)

    # Quick summary
    widths = [np.linalg.norm(left[i]-right[j]) for (i,j) in pairs]
    w_mean = float(np.mean(widths))
    w_min  = float(np.min(widths))
    w_max  = float(np.max(widths))
    print(f"Pairs: {len(pairs)} | width mean={w_mean:.2f} m (min={w_min:.2f}, max={w_max:.2f})")
    print(f"Centerline points: {len(mids_ord)} -> {outp}")

if __name__ == "__main__":
    main()
