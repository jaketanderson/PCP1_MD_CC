#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os

# Keep each worker's BLAS single-threaded so that running many conditions in
# parallel doesn't oversubscribe the CPUs. Must be set before numpy is imported
# (directly or via MDAnalysis/pandas). setdefault lets the caller override.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # headless / process-pool safe; we only ever save figures
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import pandas as pd
import numpy as np
from tqdm import tqdm


# In[2]:

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


# In[3]:


BURN_IN_NS = 50
REPORT_INTERVAL_PS = 20
BURN_IN_FRAMES = int(BURN_IN_NS * 1000 / REPORT_INTERVAL_PS)
BURN_IN_PS = BURN_IN_NS * 1000
RESIDUE_OFFSET = 1400
N_CORE_RESIDUES = 8


def condition_labels(path):
    """Return readable (state, ff, water_model) labels for a condition or
    replicate path, e.g. 'apo/ff14sb/opc' or 'apo/ff14sb/opc/0'."""
    state, ff, water_model = path.split("/")[:3]
    ff_label = ff[:4] + ff[4:].upper()   # ff14sb -> ff14SB
    water_label = water_model.upper()    # opc -> OPC, tip3p -> TIP3P
    return state, ff_label, water_label


# ### Trim away our 50ns (dt=10ps) burn-in

# In[4]:


def create_trimmed(prefixes, data):
    for prefix in prefixes:
        u_full = mda.Universe(f"../workspace/{prefix}/minimized.pdb", f"../workspace/{prefix}/production.dcd")

        with mda.Writer(f"../workspace/{prefix}/production_trimmed.dcd", u_full.trajectory.n_atoms, dt=u_full.trajectory.time, istart=1) as W:
            for ts in u_full.trajectory[BURN_IN_FRAMES:]:
                W.write(u_full.atoms)

        u_trimmed = mda.Universe(f"../workspace/{prefix}/minimized.pdb", f"../workspace/{prefix}/production_trimmed.dcd")
        data[prefix]["u_trimmed"] = u_trimmed

        df = pd.read_csv(f"../workspace/{prefix}/production.log")
        data[prefix]["df_trimmed"] = df[df["Time (ps)"] >= BURN_IN_PS]

    return


# In[5]:


def plot_timeseries(prefixes, data):
    for prefix in prefixes:
        df_trimmed = data[prefix]["df_trimmed"]

        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 9))

        colors = {"Kinetic Energy (kJ/mole)": "tab:orange",
                  "Temperature (K)": "tab:red",
                  "Potential Energy (kJ/mole)": "tab:blue",
                  "Total Energy (kJ/mole)": "tab:green",
                 }

        for col in ["Kinetic Energy (kJ/mole)", "Temperature (K)"]:
            axes[0].plot(df_trimmed["Time (ps)"] - BURN_IN_PS,
                         df_trimmed[col],
                         color=colors[col],
                         label=col+"\n"+r"$\mu=$"+f"{np.mean(df_trimmed[col]):0.2e}"+r", $\sigma=$" + f"{np.std(df_trimmed[col]):0.2e}")
        axes[0].legend()
        axes[0].grid(axis="x")
        axes[0].set_title(f"Timeseries from ../workspace/{prefix}/production_trimmed.log")

        for col in ["Potential Energy (kJ/mole)", "Total Energy (kJ/mole)"]:
            axes[1].plot(df_trimmed["Time (ps)"] - BURN_IN_PS,
                         df_trimmed[col],
                         color=colors[col],
                         label=col+"\n"+r"$\mu=$"+f"{np.mean(df_trimmed[col]):0.2e}"+r", $\sigma=$" + f"{np.std(df_trimmed[col]):0.2e}")
        axes[1].legend()
        axes[1].grid(axis="x")

        plt.xlabel("Time (ns)")
        plt.xlim(0, max(df_trimmed["Time (ps)"] - BURN_IN_PS))
        xticks = np.arange(0, 500_001, 1e5)
        plt.xticks(xticks, labels=[int(x/1000) for x in xticks])
        plt.tight_layout()
        fig.savefig(f"{prefix}/timeseries.png")
        plt.close()

    return


# $$DCC(i,j) = \frac
# {\left< \Delta\mathbf{r}_i(t) \cdot \Delta\mathbf{r}_j(t) \right>_t}
# {\sqrt{\left< \| \Delta\mathbf{r}_i(t) \|^2 \right>_t} \sqrt{\left< \| \Delta\mathbf{r}_j(t) \|^2 \right>_t}
# }$$
# $$
# \Delta\mathbf{r}_i(t) = \mathbf{r}_i(t) - \left< \mathbf{r}_i(t)\right>_t
# $$

# In[6]:


def prepare_for_DCC(prefixes, data):
    for prefix in prefixes:

        # Remove overall translations and rotations
        ref = mda.Universe(f"../workspace/{prefix}/minimized.pdb")
        u_trimmed = data[prefix]["u_trimmed"]
        u_trimmed.trajectory[0]
        ref.atoms.positions = u_trimmed.atoms.positions.copy()

        align.AlignTraj(
            u_trimmed, ref,
            select="protein and name CA",
            in_memory=True,
            match_atoms=True,
        ).run()

        ca = u_trimmed.select_atoms("protein and name CA")
        rmsf = rms.RMSF(ca).run()
        rmsf_values = rmsf.rmsf
        data[prefix]["rmsf"] = rmsf_values
        core_mask = np.argsort(rmsf_values)[:N_CORE_RESIDUES]
        core_resids = ca.resids[core_mask]
        core_sel = "protein and name CA and resid " + " ".join(map(str, core_resids))

        align.AlignTraj(
            u_trimmed, ref,
            select=core_sel,
            in_memory=True,
            match_atoms=True,
        ).run()

        with mda.Writer(f"../workspace/{prefix}/production_trimmed_static.dcd", u_trimmed.trajectory.n_atoms, dt=u_trimmed.trajectory.time, istart=1) as W:
            for ts in u_trimmed.trajectory:
                W.write(u_trimmed.atoms)

        # Create an object that stores the 3D position for every Ca for every frame
        ca = u_trimmed.select_atoms("protein and name CA")
        data[prefix]["ca"] = ca
        ca_pos = np.empty((u_trimmed.trajectory.n_frames, ca.n_atoms, 3), dtype=np.float32)
        for i, ts in enumerate(u_trimmed.trajectory):
            ca_pos[i] = ca.positions
        data[prefix]["ca_pos"] = ca_pos

    return


# In[7]:


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


def create_DCC_matrices(prefixes, data):
    for prefix in prefixes:
        ca = data[prefix]["ca"]
        ca_pos = data[prefix]["ca_pos"]
        T = ca_pos.shape[0]

        # Vectorized DCC: delta[t,i,xyz] = pos - mean_pos
        delta = ca_pos - ca_pos.mean(axis=0)
        cov = np.einsum('tix,tjx->ij', delta, delta, optimize=True) / T
        std = np.sqrt(np.diag(cov))
        DCC_matrix = cov / np.outer(std, std)
        data[prefix]["DCC_matrix"] = DCC_matrix

        # Average distance matrix (chunked over time to bound memory)
        data[prefix]["avg_dist_matrix"] = _avg_distance_matrix(ca_pos)

    return


# In[8]:


def _plot_dual_heatmap(DCC_matrix, avg_dist_matrix, n_atoms, dcc_title, dist_title, save_path):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes()
    heatmap = ax.imshow(DCC_matrix, origin='lower', cmap="bwr", vmin=-1.0, vmax=1.0)

    cax = fig.add_axes([ax.get_position().x1+0.04, ax.get_position().y0, 0.08, ax.get_position().height])
    plt.colorbar(heatmap, cax=cax)
    ax.set_xticks(np.arange(0, n_atoms, 10))
    ax.set_yticks(np.arange(0, n_atoms, 10))
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    ax.set_title(dcc_title)

    ax2 = fig.add_axes([cax.get_position().x1+0.04, cax.get_position().y0, 1.2, cax.get_position().height])
    heatmap2 = ax2.imshow(avg_dist_matrix, origin='lower', cmap="plasma")

    cax2 = fig.add_axes([ax2.get_position().x1+0.04, ax2.get_position().y0, 0.08, ax2.get_position().height])
    plt.colorbar(heatmap2, cax=cax2)
    ax2.set_xticks(np.arange(0, n_atoms, 10))
    ax2.set_yticks(np.arange(0, n_atoms, 10))
    ax2.set_xlabel("Residue index")
    ax2.set_ylabel("Residue index")
    ax2.set_title(dist_title)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close()


# In[9]:


def create_individual_heatmaps(prefixes, data):
    for prefix in prefixes:
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


# In[10]:


def create_averaged_heatmap(prefixes, data):
    condition_dir = os.path.dirname(prefixes[0])
    state, ff_label, water_label = condition_labels(condition_dir)
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


# In[11]:


# min_dist = 10
# filtered_DCC_matrix = np.where(avg_dist_matrix >= min_dist, DCC_matrix, np.nan)

# fig = plt.figure(figsize=(6,6))
# ax = plt.axes()
# ax.imshow(np.ones(shape=DCC_matrix.shape)*0.5, cmap="Greys", vmin=0, vmax=1)
# heatmap = ax.imshow(filtered_DCC_matrix, origin='lower', cmap="bwr", vmin=-1.0, vmax=1.0)

# cax = fig.add_axes([ax.get_position().x1+0.04,ax.get_position().y0,0.08,ax.get_position().height])
# plt.colorbar(heatmap, cax=cax)
# ax.set_xticks(np.arange(0, ca.n_atoms, 10))
# ax.set_yticks(np.arange(0, ca.n_atoms, 10))
# ax.set_xlabel("Residue index")
# ax.set_ylabel("Residue index")
# ax.set_title(f"Dynamic cross-correlation matrix\nof $\\alpha$-carbons in apo-PCP1\n{min_dist}$\\AA$ minimum avg. distance threshold")

# plt.show()
# plt.close()


# In[12]:


def create_RMSF_hists(prefixes, data):
    ymax = max(max(data[prefix]["rmsf"]) for prefix in prefixes)

    for prefix in prefixes:
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
        plt.close()


# In[13]:


def create_averaged_RMSF_hist(prefixes, data):
    condition_dir = os.path.dirname(prefixes[0])
    state, ff_label, water_label = condition_labels(condition_dir)
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
    plt.title(r"Root mean square fluctuations of " + f"{state}-PCP1 ({ff_label}, {water_label}) " + r"$\alpha$-carbons" + f"\naveraged from {len(prefixes)} replicates")
    plt.savefig(f"{condition_dir}/average_RMSF_hist.png")
    plt.close()


# In[14]:


def process_condition(prefix):
    """Run the full analysis pipeline for one condition's replicates.

    Self-contained so it can run in its own process: owns its output dirs and
    its own `data` dict (nothing is shared across workers).
    """
    replicates = [f"{prefix}/{replicate_id}" for replicate_id in replicate_ids]
    for replicate in replicates:
        os.makedirs(replicate, exist_ok=True)
    data = {replicate: {} for replicate in replicates}

    create_trimmed(replicates, data)
    plot_timeseries(replicates, data)
    prepare_for_DCC(replicates, data)
    create_DCC_matrices(replicates, data)
    create_individual_heatmaps(replicates, data)
    create_averaged_heatmap(replicates, data)
    create_RMSF_hists(replicates, data)
    create_averaged_RMSF_hist(replicates, data)
    return prefix


# Peak resident memory per worker: a condition holds its 4 replicate
# trajectories in memory at once (~3 GB each via in_memory alignment) plus small
# NxN matrices. Tune if your systems are much larger/smaller.
MEM_PER_WORKER_GB = 13


def default_workers():
    """Pick a worker count bounded by both CPUs and available RAM, so a big
    box doesn't OOM by launching one worker per core."""
    cpu = os.cpu_count() or 1
    try:
        total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        mem_cap = int(total_gb * 0.8 / MEM_PER_WORKER_GB)
    except (ValueError, OSError, AttributeError):
        mem_cap = cpu  # sysconf unavailable (e.g. Windows) -> fall back to CPUs
    return max(1, min(len(prefixes), cpu, mem_cap))


if __name__ == "__main__":
    # Conditions are independent and write to disjoint directories, so we run
    # them across a process pool. Override the worker count with N_WORKERS.
    n_workers = int(os.environ.get("N_WORKERS", default_workers()))
    print(f"Processing {len(prefixes)} conditions across {n_workers} worker(s)")

    failures = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_condition, prefix): prefix for prefix in prefixes}
        for future in tqdm(as_completed(futures), total=len(futures)):
            prefix = futures[future]
            try:
                future.result()
            except Exception:
                failures.append(prefix)
                print(f"\n[FAILED] {prefix}\n{traceback.format_exc()}")

    if failures:
        print(f"\n{len(failures)} condition(s) failed: {failures}")
    else:
        print("\nAll conditions completed successfully")
