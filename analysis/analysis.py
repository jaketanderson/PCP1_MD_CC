#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------
# Performance changes vs. the original script:
#
# 1. create_trimmed() no longer writes the trimmed trajectory to disk and
#    then re-opens it from disk. It uses Universe.transfer_to_memory(start=...)
#    to slice off the burn-in directly into RAM, eliminating one full read
#    pass over the (potentially huge) trajectory per replicate.
# 2. create_trimmed() and prepare_for_DCC() now process a condition's 4
#    replicates concurrently via ThreadPoolExecutor, the same pattern the
#    original script already used for create_DCC_matrices(). Since a
#    condition's 4 replicates are already held in memory simultaneously
#    (that's how MEM_PER_WORKER_GB was budgeted), this doesn't raise peak
#    memory -- it just uses CPU that was previously left idle.
# 3. prepare_for_DCC() builds ca_pos via a single vectorized slice of the
#    in-memory trajectory's coordinate array instead of a Python
#    "for ts in trajectory" loop.
# 4. Two module-level flags (WRITE_TRIMMED_DCD / WRITE_STATIC_DCD) let you
#    skip writing production_trimmed.dcd / production_trimmed_static.dcd to
#    disk entirely if nothing outside this script reads them.
# 5. *_NUM_THREADS (OMP/OPENBLAS/MKL) are now sized automatically from the
#    actual N_WORKERS this run ends up using, instead of a hardcoded "1".
#    See the block right below -- it fills whatever CPU headroom is left
#    after N_WORKERS processes x 4 replicate-threads without oversubscribing.
#
# Plotting functions (plot_timeseries, heatmaps, RMSF hists) are left
# sequential on purpose: they use matplotlib.pyplot's global current-figure
# state (plt.figure()/plt.axes()), which is not safe to call concurrently
# from multiple threads. Threading those would risk corrupted/crossed plots.
# Note this also means, during the plotting stages, only N_WORKERS cores are
# in use (not N_WORKERS*4) -- BLAS thread tuning doesn't touch that idle
# time, since matplotlib rendering isn't BLAS work. If plotting turns out to
# be a meaningful chunk of your runtime, the real fix would be rewriting
# those functions against matplotlib's Figure API directly (instead of
# pyplot's global state) so they can be threaded safely too -- happy to do
# that if it turns out to matter for you.
# ---------------------------------------------------------------------------

import os
import sys
import time
import logging
import traceback

# --- Work out condition/replicate counts and the process/thread budget
# BEFORE importing numpy or anything BLAS-backed (matplotlib, pandas,
# MDAnalysis all pull in numpy), since OMP/OPENBLAS/MKL_NUM_THREADS only
# take effect if set before those libraries are first imported. This block
# intentionally uses only the standard library.

states = ("apo", "holo", "cys-loaded")
ffs = ("ff14sb", "ff19sb")
water_models = ("opc", "tip3p")
replicate_ids = (0, 1, 2, 3)

prefixes = [
    f"{state}/{ff}/{water_model}"
    for state in states
    for ff in ffs
    for water_model in water_models
]

# Threads spawned per condition-process during create_trimmed/
# prepare_for_DCC/create_DCC_matrices (one per replicate).
REPLICATES_PER_CONDITION = len(replicate_ids)

# Peak resident memory per worker: a condition holds its 4 replicate
# trajectories in memory at once (~3 GB each via in_memory alignment) plus small
# NxN matrices. Tune if your systems are much larger/smaller.
MEM_PER_WORKER_GB = 20


def default_workers():
    """Pick a worker count bounded by CPUs, available RAM, and the number of
    conditions (there's nothing to parallelize past 12 conditions)."""
    cpu = os.cpu_count() or 1
    try:
        total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        mem_cap = int(total_gb * 0.8 / MEM_PER_WORKER_GB)
    except (ValueError, OSError, AttributeError):
        mem_cap = cpu  # sysconf unavailable (e.g. Windows) -> fall back to CPUs
    return max(1, min(len(prefixes), cpu, mem_cap))


N_WORKERS = int(os.environ.get("N_WORKERS", default_workers()))

# Peak concurrent execution contexts during the threaded stages is
# N_WORKERS * REPLICATES_PER_CONDITION. Give each one a BLAS thread budget
# that fills whatever's left of the box without going past it. Only a
# couple of BLAS-backed calls actually use multiple threads here (mainly the
# einsum in create_DCC_matrices -- MDAnalysis's RMSD fitting via qcprot is
# not BLAS-threaded), and very high per-call thread counts rarely help on
# matrices this small, hence the cap at 8. os.environ.setdefault means an
# explicit `OMP_NUM_THREADS=N python ...` from you still wins.
_cpu_count = os.cpu_count() or 1
THREADS_PER_WORKER = max(1, min(8, _cpu_count // max(1, N_WORKERS * REPLICATES_PER_CONDITION)))

os.environ.setdefault("OMP_NUM_THREADS", str(THREADS_PER_WORKER))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS_PER_WORKER))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS_PER_WORKER))

import matplotlib
matplotlib.use("Agg")  # headless / process-pool safe; we only ever save figures
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import pandas as pd
import numpy as np
from tqdm import tqdm


# ----------------------------
# Logging setup
# ----------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(processName)s | %(threadName)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class log_timer:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.t0 = time.perf_counter()
        logger.info("%s ...", self.msg)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            logger.info("%s done in %.2fs", self.msg, dt)
        else:
            logger.exception("%s failed after %.2fs", self.msg, dt)
        return False


BURN_IN_NS = 50
REPORT_INTERVAL_PS = 20
BURN_IN_FRAMES = int(BURN_IN_NS * 1000 / REPORT_INTERVAL_PS)
BURN_IN_PS = BURN_IN_NS * 1000
RESIDUE_OFFSET = 1400
N_CORE_RESIDUES = 8

# Set either of these to False if nothing outside this script reads the
# corresponding intermediate trajectory file -- skipping the write avoids a
# full disk write of a (potentially huge) trajectory per replicate.
WRITE_TRIMMED_DCD = True
WRITE_STATIC_DCD = True


def condition_labels(path):
    """Return readable (state, ff, water_model) labels for a condition or
    replicate path, e.g. 'apo/ff14sb/opc' or 'apo/ff14sb/opc/0'."""
    state, ff, water_model = path.split("/")[:3]
    ff_label = ff[:4] + ff[4:].upper()   # ff14sb -> ff14SB
    water_label = water_model.upper()    # opc -> OPC, tip3p -> TIP3P
    return state, ff_label, water_label


# ---------------------------------------------------------------------------
# Trim away our 50ns (dt=10ps) burn-in
# ---------------------------------------------------------------------------

def _trim_one(prefix):
    """Load one replicate and slice off the burn-in, entirely in memory."""
    logger.info("[%s] trimming replicate: loading trajectory", prefix)
    u_full = mda.Universe(f"../workspace/{prefix}/minimized.pdb", f"../workspace/{prefix}/production.dcd")

    # Slice straight into memory instead of writing a trimmed DCD to disk and
    # then re-opening it as a new Universe -- that was a full extra write
    # plus a full extra read of the whole (post-burn-in) trajectory.
    logger.info("[%s] trimming burn-in from frame %d", prefix, BURN_IN_FRAMES)
    u_full.transfer_to_memory(start=BURN_IN_FRAMES)

    if WRITE_TRIMMED_DCD:
        logger.info("[%s] writing production_trimmed.dcd", prefix)
        with mda.Writer(
            f"../workspace/{prefix}/production_trimmed.dcd",
            u_full.trajectory.n_atoms,
            dt=u_full.trajectory.time,
            istart=1,
        ) as W:
            for ts in u_full.trajectory:
                W.write(u_full.atoms)

    logger.info("[%s] reading production.log", prefix)
    df = pd.read_csv(f"../workspace/{prefix}/production.log")
    df_trimmed = df[df["Time (ps)"] >= BURN_IN_PS]

    logger.info("[%s] trimmed log rows: %d -> %d", prefix, len(df), len(df_trimmed))
    return prefix, u_full, df_trimmed


def create_trimmed(prefixes, data):
    logger.info("Starting trim stage for %d replicates", len(prefixes))
    max_workers = min(len(prefixes), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_trim_one, prefix): prefix for prefix in prefixes}
        for future in as_completed(futures):
            prefix = futures[future]
            try:
                prefix, u_trimmed, df_trimmed = future.result()
                data[prefix]["u_trimmed"] = u_trimmed
                data[prefix]["df_trimmed"] = df_trimmed
                logger.info("[%s] trim stage complete", prefix)
            except Exception:
                logger.exception("[%s] trim stage failed", prefix)
                raise


def plot_timeseries(prefixes, data):
    logger.info("Starting timeseries plotting for %d replicates", len(prefixes))
    for prefix in prefixes:
        logger.info("[%s] plotting timeseries", prefix)
        df_trimmed = data[prefix]["df_trimmed"]

        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 9))

        colors = {
            "Kinetic Energy (kJ/mole)": "tab:orange",
            "Temperature (K)": "tab:red",
            "Potential Energy (kJ/mole)": "tab:blue",
            "Total Energy (kJ/mole)": "tab:green",
        }

        for col in ["Kinetic Energy (kJ/mole)", "Temperature (K)"]:
            axes[0].plot(
                df_trimmed["Time (ps)"] - BURN_IN_PS,
                df_trimmed[col],
                color=colors[col],
                label=col + "\n" + r"$\mu=$" + f"{np.mean(df_trimmed[col]):0.2e}" + r", $\sigma=$" + f"{np.std(df_trimmed[col]):0.2e}",
            )
        axes[0].legend()
        axes[0].grid(axis="x")
        axes[0].set_title(f"Timeseries from ../workspace/{prefix}/production_trimmed.log")

        for col in ["Potential Energy (kJ/mole)", "Total Energy (kJ/mole)"]:
            axes[1].plot(
                df_trimmed["Time (ps)"] - BURN_IN_PS,
                df_trimmed[col],
                color=colors[col],
                label=col + "\n" + r"$\mu=$" + f"{np.mean(df_trimmed[col]):0.2e}" + r", $\sigma=$" + f"{np.std(df_trimmed[col]):0.2e}",
            )
        axes[1].legend()
        axes[1].grid(axis="x")

        plt.xlabel("Time (ns)")
        plt.xlim(0, max(df_trimmed["Time (ps)"] - BURN_IN_PS))
        xticks = np.arange(0, 500_001, 1e5)
        plt.xticks(xticks, labels=[int(x / 1000) for x in xticks])
        plt.tight_layout()
        fig.savefig(f"{prefix}/timeseries.png")
        plt.close(fig)
        logger.info("[%s] timeseries saved", prefix)


# $$DCC(i,j) = \frac
# {\left< \Delta\mathbf{r}_i(t) \cdot \Delta\mathbf{r}_j(t) \right>_t}
# {\sqrt{\left< \| \Delta\mathbf{r}_i(t) \|^2 \right>_t} \sqrt{\left< \| \Delta\mathbf{r}_j(t) \|^2 \right>_t}
# }$$
# $$
# \Delta\mathbf{r}_i(t) = \mathbf{r}_i(t) - \left< \mathbf{r}_i(t)\right>_t
# $$

def _prepare_one(prefix, u_trimmed):
    """Align one replicate and extract its CA time series."""
    logger.info("[%s] preparing for DCC: loading reference structure", prefix)
    ref = mda.Universe(f"../workspace/{prefix}/minimized.pdb")

    u_trimmed.trajectory[0]
    ref.atoms.positions = u_trimmed.atoms.positions.copy()

    logger.info("[%s] running first alignment", prefix)
    align.AlignTraj(
        u_trimmed, ref,
        select="protein and name CA",
        in_memory=True,
        match_atoms=True,
    ).run()

    ca = u_trimmed.select_atoms("protein and name CA")
    logger.info("[%s] computing RMSF for %d CA atoms", prefix, ca.n_atoms)
    rmsf = rms.RMSF(ca).run()
    rmsf_values = rmsf.rmsf
    core_mask = np.argsort(rmsf_values)[:N_CORE_RESIDUES]
    core_resids = ca.resids[core_mask]
    core_sel = "protein and name CA and resid " + " ".join(map(str, core_resids))

    logger.info("[%s] running second alignment on core residues", prefix)
    align.AlignTraj(
        u_trimmed, ref,
        select=core_sel,
        in_memory=True,
        match_atoms=True,
    ).run()

    if WRITE_STATIC_DCD:
        logger.info("[%s] writing production_trimmed_static.dcd", prefix)
        with mda.Writer(
            f"../workspace/{prefix}/production_trimmed_static.dcd",
            u_trimmed.trajectory.n_atoms,
            dt=u_trimmed.trajectory.time,
            istart=1,
        ) as W:
            for ts in u_trimmed.trajectory:
                W.write(u_trimmed.atoms)

    # Create an object that stores the 3D position for every Ca for every frame.
    # AlignTraj(..., in_memory=True) has already converted u_trimmed's
    # trajectory into a MemoryReader backed by a single (n_frames, n_atoms, 3)
    # numpy array, so we can pull every frame's CA coordinates with one
    # vectorized slice instead of looping "for ts in trajectory" in Python.
    ca = u_trimmed.select_atoms("protein and name CA")
    logger.info("[%s] extracting CA coordinate array: %d frames x %d atoms", prefix, u_trimmed.trajectory.n_frames, ca.n_atoms)
    ca_pos = u_trimmed.trajectory.coordinate_array[:, ca.indices, :].astype(np.float32, copy=True)

    return prefix, ca, ca_pos, rmsf_values


def prepare_for_DCC(prefixes, data):
    logger.info("Starting DCC preparation for %d replicates", len(prefixes))
    max_workers = min(len(prefixes), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_prepare_one, prefix, data[prefix]["u_trimmed"])
            for prefix in prefixes
        ]
        for future in as_completed(futures):
            try:
                prefix, ca, ca_pos, rmsf_values = future.result()
                data[prefix]["ca"] = ca
                data[prefix]["ca_pos"] = ca_pos
                data[prefix]["rmsf"] = rmsf_values
                logger.info("[%s] DCC prep complete", prefix)
            except Exception:
                logger.exception("DCC prep failed for a replicate")
                raise


def _avg_distance_matrix(ca_pos, chunk=500):
    """Mean over time of the pairwise CA-CA distance matrix.

    Accumulates in time-chunks so the temporary (chunk, N, N, 3) array stays
    small; the naive (T, N, N, 3) broadcast is multiple GB for a full
    trajectory and would OOM when many conditions run in parallel.
    """
    T, N, _ = ca_pos.shape
    acc = np.zeros((N, N), dtype=np.float64)
    for start in range(0, T, chunk):
        block = ca_pos[start:start + chunk]  # (c, N, 3)
        diff = block[:, :, np.newaxis, :] - block[:, np.newaxis, :, :]
        acc += np.linalg.norm(diff, axis=-1).sum(axis=0)
    return (acc / T).astype(np.float32)


def _dcc_and_avg_dist(ca_pos):
    T = ca_pos.shape[0]

    # Vectorized DCC: delta[t,i,xyz] = pos - mean_pos
    delta = ca_pos - ca_pos.mean(axis=0)
    cov = np.einsum("tix,tjx->ij", delta, delta, optimize=True) / T
    std = np.sqrt(np.diag(cov))
    DCC_matrix = cov / np.outer(std, std)

    # Average distance matrix (chunked over time to bound memory)
    avg_dist_matrix = _avg_distance_matrix(ca_pos)

    return DCC_matrix, avg_dist_matrix


def create_DCC_matrices(prefixes, data):
    """Compute the DCC and average-distance matrices for each replicate.

    Replicates are independent, and the heavy numpy work here (einsum,
    broadcast norms) releases the GIL, so we fan them out across threads.
    The outer process-per-condition pool is memory-bound (see
    MEM_PER_WORKER_GB), not CPU-bound, which normally leaves cores idle
    while a condition's 4 replicates are crunched one at a time.
    """
    logger.info("Starting DCC matrix computation for %d replicates", len(prefixes))
    max_workers = min(len(prefixes), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_dcc_and_avg_dist, data[prefix]["ca_pos"]): prefix for prefix in prefixes}
        for future in as_completed(futures):
            prefix = futures[future]
            try:
                DCC_matrix, avg_dist_matrix = future.result()
                data[prefix]["DCC_matrix"] = DCC_matrix
                data[prefix]["avg_dist_matrix"] = avg_dist_matrix
                logger.info("[%s] DCC matrices complete", prefix)
            except Exception:
                logger.exception("[%s] DCC matrix computation failed", prefix)
                raise


def _plot_dual_heatmap(DCC_matrix, avg_dist_matrix, n_atoms, dcc_title, dist_title, save_path):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes()
    heatmap = ax.imshow(DCC_matrix, origin="lower", cmap="bwr", vmin=-1.0, vmax=1.0)

    cax = fig.add_axes([ax.get_position().x1 + 0.04, ax.get_position().y0, 0.08, ax.get_position().height])
    plt.colorbar(heatmap, cax=cax)
    ax.set_xticks(np.arange(0, n_atoms, 10))
    ax.set_yticks(np.arange(0, n_atoms, 10))
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    ax.set_title(dcc_title)

    ax2 = fig.add_axes([cax.get_position().x1 + 0.04, cax.get_position().y0, 1.2, cax.get_position().height])
    heatmap2 = ax2.imshow(avg_dist_matrix, origin="lower", cmap="plasma")

    cax2 = fig.add_axes([ax2.get_position().x1 + 0.04, ax2.get_position().y0, 0.08, ax2.get_position().height])
    plt.colorbar(heatmap2, cax=cax2)
    ax2.set_xticks(np.arange(0, n_atoms, 10))
    ax2.set_yticks(np.arange(0, n_atoms, 10))
    ax2.set_xlabel("Residue index")
    ax2.set_ylabel("Residue index")
    ax2.set_title(dist_title)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def create_individual_heatmaps(prefixes, data):
    logger.info("Starting individual heatmaps for %d replicates", len(prefixes))
    for prefix in prefixes:
        logger.info("[%s] creating heatmap", prefix)
        state, ff_label, water_label = condition_labels(prefix)
        n_atoms = data[prefix]["ca"].n_atoms
        _plot_dual_heatmap(
            data[prefix]["DCC_matrix"],
            data[prefix]["avg_dist_matrix"],
            n_atoms,
            f"Dynamic cross-correlation matrix\nof $\\alpha$-carbons in {state}-PCP1 ({ff_label}, {water_label})\n" + f"from {prefix}",
            f"Time-average distance ($\\AA$) matrix\nof $\\alpha$-carbons in {state}-PCP1 ({ff_label}, {water_label})" + f"\nfrom {prefix}",
            f"{prefix}/heatmap.png",
        )
        logger.info("[%s] heatmap saved", prefix)


def create_averaged_heatmap(prefixes, data):
    condition_dir = os.path.dirname(prefixes[0])
    state, ff_label, water_label = condition_labels(condition_dir)
    logger.info("[%s] creating averaged heatmap from %d replicates", condition_dir, len(prefixes))
    DCC_matrix = np.mean([data[prefix]["DCC_matrix"] for prefix in prefixes], axis=0)
    np.save(f"{condition_dir}/avg_DCC_matrix.npy", DCC_matrix)
    avg_dist_matrix = np.mean([data[prefix]["avg_dist_matrix"] for prefix in prefixes], axis=0)
    np.save(f"{condition_dir}/avg_dist_matrix.npy", avg_dist_matrix)
    n_atoms = data[prefixes[0]]["ca"].n_atoms

    _plot_dual_heatmap(
        DCC_matrix,
        avg_dist_matrix,
        n_atoms,
        f"Dynamic cross-correlation matrix\nof $\\alpha$-carbons in {state}-PCP1 ({ff_label}, {water_label})\n" + f"averaged from {len(prefixes)} replicates",
        f"Time-average distance ($\\AA$) matrix\nof $\\alpha$-carbons in {state}-PCP1 ({ff_label}, {water_label})" + f"\naveraged from {len(prefixes)} replicates",
        f"{condition_dir}/average_heatmap.png",
    )
    logger.info("[%s] averaged heatmap saved", condition_dir)


def create_RMSF_hists(prefixes, data):
    logger.info("Starting RMSF histograms for %d replicates", len(prefixes))
    ymax = max(max(data[prefix]["rmsf"]) for prefix in prefixes)

    for prefix in prefixes:
        logger.info("[%s] creating RMSF histogram", prefix)
        state, ff_label, water_label = condition_labels(prefix)
        ca = data[prefix]["ca"]
        fig = plt.figure(figsize=(12, 6))
        plt.bar(np.arange(0, ca.n_atoms) + RESIDUE_OFFSET, data[prefix]["rmsf"])
        plt.xlabel("Residue index")
        plt.xlim(-1 + RESIDUE_OFFSET, ca.n_atoms + RESIDUE_OFFSET)
        plt.ylabel(r"RMSF ($\AA$)")
        plt.ylim(0, ymax + 1)
        plt.title(r"Root mean square fluctuations of $\alpha$-carbons" + f"\nin {state}-PCP1 ({ff_label}, {water_label}) from {prefix}")
        plt.savefig(f"{prefix}/RMSF_hist.png")
        plt.close(fig)
        logger.info("[%s] RMSF histogram saved", prefix)


def create_averaged_RMSF_hist(prefixes, data):
    condition_dir = os.path.dirname(prefixes[0])
    state, ff_label, water_label = condition_labels(condition_dir)
    logger.info("[%s] creating averaged RMSF histogram from %d replicates", condition_dir, len(prefixes))
    R = [data[prefix]["rmsf"] for prefix in prefixes]
    ca = data[prefixes[0]]["ca"]
    R_avg = np.mean(R, axis=0)
    np.save(f"{condition_dir}/R_avg.npy", R_avg, allow_pickle=False)
    R_sem = np.std(R, axis=0) / np.sqrt(len(prefixes))
    fig = plt.figure(figsize=(12, 6))
    plt.bar(np.arange(0, ca.n_atoms) + RESIDUE_OFFSET, R_avg, yerr=R_sem)
    plt.xlabel("Residue index")
    plt.xlim(-1 + RESIDUE_OFFSET, ca.n_atoms + RESIDUE_OFFSET)
    plt.ylabel(r"RMSF ($\AA$)")
    plt.title(
        r"Root mean square fluctuations of "
        + f"{state}-PCP1 ({ff_label}, {water_label}) "
        + r"$\alpha$-carbons"
        + f"\naveraged from {len(prefixes)} replicates"
    )
    plt.savefig(f"{condition_dir}/average_RMSF_hist.png")
    plt.close(fig)
    logger.info("[%s] averaged RMSF histogram saved", condition_dir)


def process_condition(prefix):
    """Run the full analysis pipeline for one condition's replicates.

    Self-contained so it can run in its own process: owns its output dirs and
    its own `data` dict (nothing is shared across workers).
    """
    logger.info("[%s] starting condition", prefix)
    with log_timer(f"[{prefix}] full condition pipeline"):
        replicates = [f"{prefix}/{replicate_id}" for replicate_id in replicate_ids]
        for replicate in replicates:
            os.makedirs(replicate, exist_ok=True)
        data = {replicate: {} for replicate in replicates}

        with log_timer(f"[{prefix}] trim stage"):
            create_trimmed(replicates, data)

        with log_timer(f"[{prefix}] timeseries plot stage"):
            plot_timeseries(replicates, data)

        with log_timer(f"[{prefix}] DCC prep stage"):
            prepare_for_DCC(replicates, data)

        with log_timer(f"[{prefix}] DCC matrix stage"):
            create_DCC_matrices(replicates, data)

        with log_timer(f"[{prefix}] individual heatmaps stage"):
            create_individual_heatmaps(replicates, data)

        with log_timer(f"[{prefix}] averaged heatmap stage"):
            create_averaged_heatmap(replicates, data)

        with log_timer(f"[{prefix}] RMSF hist stage"):
            create_RMSF_hists(replicates, data)

        with log_timer(f"[{prefix}] averaged RMSF stage"):
            create_averaged_RMSF_hist(replicates, data)

    logger.info("[%s] condition complete", prefix)
    return prefix


if __name__ == "__main__":
    # Conditions are independent and write to disjoint directories, so we run
    # them across a process pool. Override the worker count with N_WORKERS,
    # and any of OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS to
    # override the auto-sized BLAS thread budget above.
    logger.info(
        "Processing %d conditions across %d worker(s), %d BLAS thread(s) per worker "
        "(%d peak concurrent replicate-threads on %d CPUs)",
        len(prefixes),
        N_WORKERS,
        THREADS_PER_WORKER,
        N_WORKERS * REPLICATES_PER_CONDITION,
        _cpu_count,
    )

    failures = []
    start_all = time.perf_counter()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_condition, prefix): prefix for prefix in prefixes}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Conditions"):
            prefix = futures[future]
            try:
                future.result()
                logger.info("[%s] finished successfully", prefix)
            except Exception:
                failures.append(prefix)
                logger.exception("[%s] failed", prefix)

    elapsed = time.perf_counter() - start_all
    if failures:
        logger.error("%d condition(s) failed: %s", len(failures), failures)
    else:
        logger.info("All conditions completed successfully in %.2fs", elapsed)
