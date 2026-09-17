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
#    skip writing production_trimmed.dcd / production_trimmed_static_vacuum.dcd
#    to disk entirely if nothing outside this script reads them.
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
ffs = ["ff19sb","ff14sb"]
water_models = ["tip3p", "opc"]
replicate_ids = (1, 2, 3)

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
# trajectories in memory at once (~3 GB each via in_memory alignment), and
# AlignTraj(..., in_memory=True) in _prepare_one does a second in-memory pass
# on top of that while a replicate is being aligned, so the transient peak
# during the trim/align stage is well above the 4 x ~3 GB baseline. (Past
# that stage, u_trimmed is dropped as soon as ca_pos/rmsf are extracted, so
# this early peak -- not the later small NxN-matrix stages -- is what sets
# the per-worker budget.) Tune if your systems are much larger/smaller.
MEM_PER_WORKER_GB = 16


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
RESIDUE_OFFSET = 1402

# Alpha carbons in the four largest helices -- used as the alignment anchor
# instead of every CA (flexible termini/loops bias a whole-protein fit) or a
# per-run RMSF-derived subset. Resid numbers are each replicate's own PDB
# numbering (analysis.py processes one condition/state at a time and never
# needs cross-state resid correspondence, unlike movies.py).
#
# Selected by "name CA" rather than "protein and name CA": the
# phosphopantetheine-attachment residue is renamed PPT in holo/cys-loaded
# (SER in apo), and PPT falls outside MDAnalysis's "protein" selection
# macro, which would silently drop that one CA (resid 38, inside the
# 38-51 helix range) from holo/cys-loaded but not apo.
CORE_HELIX_RESID_RANGES = ((7, 20), (38, 51), (57, 62), (69, 72))
CORE_ALIGN_SEL = "name CA and (" + " or ".join(
    f"resid {lo}-{hi}" for lo, hi in CORE_HELIX_RESID_RANGES
) + ")"

# Set either of these to False if nothing outside this script reads the
# corresponding intermediate trajectory file -- skipping the write avoids a
# full disk write of a (potentially huge) trajectory per replicate.
WRITE_TRIMMED_DCD = True
WRITE_STATIC_DCD = True

# production_trimmed_static_vacuum.dcd: solvent/ion-stripped ("vacuum"), at the
# native REPORT_INTERVAL_PS (20 ps) timestep -- every frame, no downsampling.
NONSOLVENT_SEL = "not resname HOH and not resname NA and not resname CL"

# Every replicate of every state gets its production_trimmed_static_vacuum.dcd
# rigidly superposed (via CORE_ALIGN_SEL) onto this one shared target, so they
# all share a common orientation above the level of individual states -- e.g.
# for viewing different states' vacuum trajectories side by side. Fixed to the
# first (state, replicate) in processing order (prefixes[0]/replicate_ids[0])
# for reproducibility, rather than whichever happens to finish first when
# conditions run concurrently.
MASTER_REFERENCE = f"{prefixes[0]}/{replicate_ids[0]}"


def build_master_target(prefix=MASTER_REFERENCE):
    """Raw (pre-alignment) first-frame core-helix-CA positions of the given
    replicate's trimmed production trajectory.

    This is the single shared target every replicate/state's vacuum dcd gets
    superposed onto. Seeks directly to frame BURN_IN_FRAMES of the raw
    production.dcd rather than loading create_trimmed()'s full in-memory
    trajectory, since only one frame's positions are needed here.
    """
    logger.info("[%s] building master alignment target", prefix)
    u = mda.Universe(f"../workspace/{prefix}/minimized.pdb", f"../workspace/{prefix}/production.dcd")
    u.trajectory[BURN_IN_FRAMES]
    return u.select_atoms(CORE_ALIGN_SEL).positions.copy()


def _superpose_transform(mobile_positions, target_positions):
    """Rigid transform (rotation, mobile_com, target_com) that superposes
    mobile_positions onto target_positions by least-squares fit. Apply as
    `(positions - mobile_com) @ rotation.T + target_com`."""
    mobile_com = mobile_positions.mean(axis=0)
    target_com = target_positions.mean(axis=0)
    rotation, _ = align.rotation_matrix(mobile_positions - mobile_com, target_positions - target_com)
    return rotation, mobile_com, target_com


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

def _prepare_one(prefix, u_trimmed, master_target):
    """Align one replicate and extract its CA time series."""
    logger.info("[%s] preparing for DCC: loading reference structure", prefix)
    ref = mda.Universe(f"../workspace/{prefix}/minimized.pdb")

    u_trimmed.trajectory[0]
    ref.atoms.positions = u_trimmed.atoms.positions.copy()

    # Captured before the per-replicate AlignTraj below mutates u_trimmed in
    # place -- this raw first-frame core-helix-CA snapshot is what gets
    # superposed onto the shared master_target for the vacuum dcd, so the
    # master alignment reflects the same "first trajectory frame" semantics
    # build_master_target() uses.
    raw_core_positions = u_trimmed.select_atoms(CORE_ALIGN_SEL).positions.copy()

    logger.info("[%s] running alignment on core helix CAs", prefix)
    align.AlignTraj(
        u_trimmed, ref,
        select=CORE_ALIGN_SEL,
        in_memory=True,
        match_atoms=True,
    ).run()

    ca = u_trimmed.select_atoms("name CA")
    logger.info("[%s] computing RMSF for %d CA atoms", prefix, ca.n_atoms)
    rmsf = rms.RMSF(ca).run()
    rmsf_values = rmsf.rmsf

    if WRITE_STATIC_DCD:
        nonsolvent = u_trimmed.select_atoms(NONSOLVENT_SEL)
        logger.info(
            "[%s] writing production_trimmed_static_vacuum.dcd (%d atoms, %d frames)",
            prefix, nonsolvent.n_atoms, u_trimmed.trajectory.n_frames,
        )
        # Stripping solvent/ions drops the atom count below minimized.pdb's,
        # so the trajectory needs its own matching topology to be loadable --
        # write one from the first output frame, same as movies.py's solute.pdb.
        topology_path = f"../workspace/{prefix}/production_trimmed_static_vacuum.pdb"
        traj_path = f"../workspace/{prefix}/production_trimmed_static_vacuum.dcd"

        # Single fixed rigid-body transform superposing this replicate onto
        # master_target (shared across every state/replicate -- see
        # MASTER_REFERENCE), computed once and applied identically to every
        # frame. One transform is enough, rather than a per-frame fit: the
        # AlignTraj above has already removed each frame's wobble relative to
        # raw_core_positions, so u_trimmed's frames only differ from each
        # other by that same internal motion -- one more rotation on top of
        # all of them brings the whole replicate into the master frame.
        rotation, mobile_com, target_com = _superpose_transform(raw_core_positions, master_target)

        def _to_master(positions):
            return (positions - mobile_com) @ rotation.T + target_com

        # AtomGroup.write() (nonsolvent.write(topology_path)) leaks badly here:
        # MDAnalysis's PDBWriter keeps a live reference to the AtomGroup it's
        # given (obj.universe -> trajectory -> the full in-memory coordinate
        # array) in a reference cycle its Cython class doesn't expose to
        # Python's cyclic gc, so it's never collected -- confirmed by tracing
        # with gc.get_referrers(): the PDBWriter instance, and everything it
        # pins, survives explicit gc.collect() calls. mda.Merge() copies the
        # current positions into a brand new, independent Universe with no
        # ties back to u_trimmed, so writing from that snapshot instead
        # doesn't pin the multi-GB trajectory in memory.
        u_trimmed.trajectory[0]
        original = nonsolvent.positions.copy()
        # Positions are written master-aligned but restored immediately after
        # each write -- nonsolvent.positions is a view straight into
        # u_trimmed's backing array (not a copy), so leaving it mutated would
        # corrupt the ca_pos extraction below, which needs the replicate's
        # own (not master-aligned) frame.
        nonsolvent.positions = _to_master(original)
        mda.Merge(nonsolvent).atoms.write(topology_path)
        nonsolvent.positions = original

        with mda.Writer(traj_path, nonsolvent.n_atoms, dt=REPORT_INTERVAL_PS, istart=1) as W:
            for ts in u_trimmed.trajectory:
                original = nonsolvent.positions.copy()
                nonsolvent.positions = _to_master(original)
                W.write(nonsolvent)
                nonsolvent.positions = original

    # Create an object that stores the 3D position for every Ca for every frame.
    # AlignTraj(..., in_memory=True) has already converted u_trimmed's
    # trajectory into a MemoryReader backed by a single (n_frames, n_atoms, 3)
    # numpy array, so we can pull every frame's CA coordinates with one
    # vectorized slice instead of looping "for ts in trajectory" in Python.
    ca = u_trimmed.select_atoms("name CA")
    logger.info("[%s] extracting CA coordinate array: %d frames x %d atoms", prefix, u_trimmed.trajectory.n_frames, ca.n_atoms)
    ca_pos = u_trimmed.trajectory.coordinate_array[:, ca.indices, :].astype(np.float32, copy=True)
    n_atoms = ca.n_atoms
    # A plain array copy, not an AtomGroup -- doesn't hold a reference back
    # to the Universe, so it's safe to keep around after u_trimmed is
    # dropped. Needed to align RMSF across states whose CA count differs
    # (e.g. apo has a CA at the phosphopantetheine-attachment residue that
    # holo/cys-loaded lack, since that residue's PPT resname there falls
    # outside the "protein" selection macro).
    resids = ca.resids.copy()

    # Return only the CA count, never the AtomGroup itself -- an AtomGroup
    # holds a live reference back to its parent Universe, which would keep
    # the whole (potentially multi-GB) all-atom trajectory pinned in memory
    # for the rest of the pipeline even after ca_pos has been extracted.
    return prefix, n_atoms, ca_pos, rmsf_values, resids


def prepare_for_DCC(prefixes, data, master_target):
    logger.info("Starting DCC preparation for %d replicates", len(prefixes))
    max_workers = min(len(prefixes), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_prepare_one, prefix, data[prefix]["u_trimmed"], master_target)
            for prefix in prefixes
        ]
        for future in as_completed(futures):
            try:
                prefix, n_atoms, ca_pos, rmsf_values, resids = future.result()
                data[prefix]["n_atoms"] = n_atoms
                data[prefix]["ca_pos"] = ca_pos
                data[prefix]["rmsf"] = rmsf_values
                data[prefix]["resids"] = resids
                logger.info("[%s] DCC prep complete", prefix)
            except Exception:
                logger.exception("DCC prep failed for a replicate")
                raise
            finally:
                # The full all-atom trajectory (transfer_to_memory'd, ~GBs)
                # is only needed to get here -- everything downstream works
                # off ca_pos/n_atoms/rmsf. Drop it now instead of holding it
                # for the rest of the pipeline (heatmaps, RMSF hists, ...).
                data[prefix].pop("u_trimmed", None)


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

    # Tick positions stay at array indices (0, 10, 20, ...) -- that's what
    # imshow's pixel grid is addressed by -- but the labels are shifted by
    # RESIDUE_OFFSET so they read the same residue numbering as the RMSF
    # hists (which plot directly against np.arange(n_atoms) + RESIDUE_OFFSET).
    tick_pos = np.arange(0, n_atoms, 10)
    tick_labels = tick_pos + RESIDUE_OFFSET

    cax = fig.add_axes([ax.get_position().x1 + 0.04, ax.get_position().y0, 0.08, ax.get_position().height])
    plt.colorbar(heatmap, cax=cax)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    ax.set_title(dcc_title)

    ax2 = fig.add_axes([cax.get_position().x1 + 0.04, cax.get_position().y0, 1.2, cax.get_position().height])
    heatmap2 = ax2.imshow(avg_dist_matrix, origin="lower", cmap="plasma")

    cax2 = fig.add_axes([ax2.get_position().x1 + 0.04, ax2.get_position().y0, 0.08, ax2.get_position().height])
    plt.colorbar(heatmap2, cax=cax2)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(tick_pos)
    ax2.set_yticklabels(tick_labels)
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
        n_atoms = data[prefix]["n_atoms"]
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
    n_atoms = data[prefixes[0]]["n_atoms"]

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
        n_atoms = data[prefix]["n_atoms"]
        fig = plt.figure(figsize=(12, 6))
        plt.bar(np.arange(0, n_atoms) + RESIDUE_OFFSET, data[prefix]["rmsf"])
        plt.xlabel("Residue index")
        plt.xlim(-1 + RESIDUE_OFFSET, n_atoms + RESIDUE_OFFSET)
        plt.ylabel(r"RMSF ($\AA$)")
        plt.ylim(0, ymax)
        plt.title(r"Root mean square fluctuations of $\alpha$-carbons" + f"\nin {state}-PCP1 ({ff_label}, {water_label}) from {prefix}")
        plt.savefig(f"{prefix}/RMSF_hist.png")
        plt.close(fig)
        logger.info("[%s] RMSF histogram saved", prefix)


def create_averaged_RMSF_hist(prefixes, data):
    condition_dir = os.path.dirname(prefixes[0])
    state, ff_label, water_label = condition_labels(condition_dir)
    logger.info("[%s] creating averaged RMSF histogram from %d replicates", condition_dir, len(prefixes))
    R = [data[prefix]["rmsf"] for prefix in prefixes]
    n_atoms = data[prefixes[0]]["n_atoms"]
    R_avg = np.mean(R, axis=0)
    np.save(f"{condition_dir}/R_avg.npy", R_avg, allow_pickle=False)
    R_sem = np.std(R, axis=0) / np.sqrt(len(prefixes))
    np.save(f"{condition_dir}/R_sem.npy", R_sem, allow_pickle=False)
    # Same across all replicates of a condition (same topology) -- saved so
    # cross-state RMSF comparisons can align by actual resid instead of
    # array position, needed whenever the two states' CA counts differ.
    np.save(f"{condition_dir}/ca_resids.npy", data[prefixes[0]]["resids"], allow_pickle=False)
    fig = plt.figure(figsize=(12, 6))
    plt.bar(np.arange(0, n_atoms) + RESIDUE_OFFSET, R_avg, yerr=R_sem)
    plt.xlabel("Residue index")
    plt.xlim(-1 + RESIDUE_OFFSET, n_atoms + RESIDUE_OFFSET)
    plt.ylim(0, 6)
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


def create_RMSF_diff_hist(state_a, state_b, ff, water_model):
    """Bar plot of per-residue RMSF differences between two states sharing
    the same force field and water model, by residue index (same layout as
    create_averaged_RMSF_hist's per-condition RMSF bar plots).

    Error bars are the standard error on the difference, propagated in
    quadrature from each condition's own across-replicate SEM (R_sem.npy):
    sigma_diff = sqrt(sigma_a^2 + sigma_b^2), valid since the two states'
    replicate sets are independent.

    States can differ in which residues carry a CA (e.g. apo has one at the
    phosphopantetheine-attachment site that holo/cys-loaded lack, since that
    residue's PPT resname there falls outside the "protein" selection macro
    used to build the CA groups). Arrays are therefore aligned by actual
    resid (ca_resids.npy) before subtracting, rather than assuming array
    position corresponds to the same residue in both states; any residue
    present in only one state is dropped from the comparison.

    Reads the R_avg.npy/R_sem.npy/ca_resids.npy each condition's
    averaged-RMSF stage already wrote, so both conditions' full pipelines
    must have completed (in this run or a previous one) before this can run.
    """
    dir_a = f"{state_a}/{ff}/{water_model}"
    dir_b = f"{state_b}/{ff}/{water_model}"
    logger.info("[%s vs %s] creating RMSF difference histogram", dir_a, dir_b)

    R_a = np.load(f"{dir_a}/R_avg.npy")
    R_b = np.load(f"{dir_b}/R_avg.npy")
    sem_a = np.load(f"{dir_a}/R_sem.npy")
    sem_b = np.load(f"{dir_b}/R_sem.npy")
    resids_a = np.load(f"{dir_a}/ca_resids.npy")
    resids_b = np.load(f"{dir_b}/ca_resids.npy")

    common_resids, idx_a, idx_b = np.intersect1d(resids_a, resids_b, return_indices=True)
    n_dropped = min(len(resids_a), len(resids_b)) - len(common_resids)
    if n_dropped:
        logger.warning(
            "[%s vs %s] %d residue(s) present in only one state, excluded from the diff",
            dir_a, dir_b, n_dropped,
        )

    diff = R_a[idx_a] - R_b[idx_b]
    diff_sem = np.sqrt(sem_a[idx_a]**2 + sem_b[idx_b]**2)
    n_atoms = len(diff)

    ff_label = ff[:4] + ff[4:].upper()   # ff19sb -> ff19SB
    water_label = water_model.upper()    # opc -> OPC

    fig = plt.figure(figsize=(12, 6))
    plt.bar(np.arange(0, n_atoms) + RESIDUE_OFFSET, diff, yerr=diff_sem)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Residue index")
    plt.xlim(-1 + RESIDUE_OFFSET, n_atoms + RESIDUE_OFFSET)
    # Ticks land on multiples of 5 (e.g. ...1405, 1410...) rather than on
    # RESIDUE_OFFSET + 5*k, which would be off-multiple whenever
    # RESIDUE_OFFSET itself isn't a multiple of 5.
    first_tick = RESIDUE_OFFSET + (-RESIDUE_OFFSET) % 5
    plt.xticks(np.arange(first_tick, n_atoms + RESIDUE_OFFSET, 5))
    plt.ylim(-0.5, 1.5)
    plt.gca().set_axisbelow(True)
    plt.grid(True)
    plt.ylabel(r"RMSF difference ($\AA$), " + f"{state_a} - {state_b}")
    plt.title(
        r"$\alpha$-carbon RMSF difference"
        + f"\n{state_a}-PCP1 vs {state_b}-PCP1 ({ff_label}, {water_label})"
    )
    out_path = f"{state_a}_vs_{state_b}_{ff}_{water_model}_RMSF_diff_hist.png"
    plt.savefig(out_path)
    plt.close(fig)
    logger.info("[%s vs %s] RMSF difference histogram saved to %s", dir_a, dir_b, out_path)
    return out_path


def process_condition(prefix, master_target):
    """Run the full analysis pipeline for one condition's replicates.

    Self-contained so it can run in its own process: owns its output dirs and
    its own `data` dict (nothing is shared across workers). master_target is
    the one exception -- it's computed once up front (see __main__) and
    passed into every condition so every replicate's vacuum dcd shares the
    same alignment frame.
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
            prepare_for_DCC(replicates, data, master_target)

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

    master_target = build_master_target() if WRITE_STATIC_DCD else None

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(process_condition, prefix, master_target): prefix
            for prefix in prefixes
        }
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

    RMSF_DIFF_PAIRS = (
        ("cys-loaded", "holo"),
        ("holo", "apo"),
        ("cys-loaded", "apo"),
    )
    for state_a, state_b in RMSF_DIFF_PAIRS:
        if f"{state_a}/ff19sb/opc" in failures or f"{state_b}/ff19sb/opc" in failures:
            continue
        with log_timer(f"RMSF difference histogram ({state_a} vs {state_b}, ff19sb/opc)"):
            create_RMSF_diff_hist(state_a, state_b, "ff19sb", "opc")
