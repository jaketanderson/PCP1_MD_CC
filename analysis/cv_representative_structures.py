#!/usr/bin/env python
# coding: utf-8
"""Find representative structures near target (theta, phi) points on the
phosphopantetheine-arm CV heatmap from CV.ipynb, and mark those points on
the heatmap.

theta/phi are the spherical angles of the PD->CP8 vector (residue 38, the
phosphopantetheine-modified residue), computed the same way as CV.ipynb's
cell 2, with one fix: the three replicates' per-frame vectors are stacked
with np.concatenate(..., axis=0) (one long series of proper 3D vectors)
instead of axis=1, which mixed replicate 1/2/3's x/y/z components together
and made the resulting norm/angles meaningless.

For each state and each target point:
  - the 5 frames (across all 3 replicates) whose (theta, phi) is nearest the
    target (phi distance wrapped at 2*pi) are written as a 5-MODEL PDB.
  - the point is drawn as a colored circle on that state's CV heatmap.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import MDAnalysis as mda

ff, water_model = "ff19sb", "opc"
replicate_ids = (1, 2, 3)
N_STRUCTURES = 5

# Point order matches within each state's list; color assignment is by
# position so the same color means "point 1/2/3" consistently across both
# states' plots and filenames.
TARGETS = {
    "holo": [(1.141, 3.917), (1.437, 3.300), (1.745, 2.500)],
    "cys-loaded": [(1.141, 3.917), (1.745, 2.500), (2.391, 1.931)],
}
# Per-state, ordered to match TARGETS[state] by position.
COLORS = {
    "holo": ["white", "cyan", "lime"],
    "cys-loaded": ["white", "lime", "magenta"],
}

# theta spans [0, pi], phi spans [0, 2*pi] -- twice the range -- so phi needs
# twice as many bins as theta for each bin to cover the same angular width
# (pi/N_BINS_THETA == 2*pi/N_BINS_PHI), making bins appear square.
N_BINS_THETA = 100
N_BINS_PHI = N_BINS_THETA * 2
HIST_BINS = [N_BINS_THETA, N_BINS_PHI]


def load_theta_phi(state):
    """theta, phi, and (replicate_id, frame_index) for every frame across
    all replicates of `state`, stacked along axis=0 so each row is one
    proper 3D PD->CP8 vector (see module docstring)."""
    vecs, reps, frames = [], [], []
    for replicate_id in replicate_ids:
        prefix = f"{state}/{ff}/{water_model}/{replicate_id}"
        u = mda.Universe(
            f"../workspace/{prefix}/production_trimmed_static_vacuum.pdb",
            f"../workspace/{prefix}/production_trimmed_static_vacuum.dcd",
        )
        pd_atom = u.select_atoms("resnum 38 and name PD").atoms[0]
        cp8_atom = u.select_atoms("resnum 38 and name CP8").atoms[0]
        n = u.trajectory.n_frames
        rep_vecs = np.empty((n, 3))
        for i, ts in enumerate(u.trajectory):
            rep_vecs[i] = cp8_atom.position - pd_atom.position
        vecs.append(rep_vecs)
        reps.append(np.full(n, replicate_id))
        frames.append(np.arange(n))

    pp_vecs = np.concatenate(vecs, axis=0)
    reps = np.concatenate(reps)
    frames = np.concatenate(frames)

    rs = np.linalg.norm(pp_vecs, axis=1)
    thetas = np.arccos(pp_vecs[:, 2] / rs)
    phis = np.arctan2(pp_vecs[:, 1], pp_vecs[:, 0]) % (2 * np.pi)
    return thetas, phis, reps, frames


def nearest_frames(thetas, phis, reps, frames, target, n=N_STRUCTURES):
    """Indices of the n frames whose (theta, phi) is closest to target,
    wrapping phi at 2*pi (theta is a bounded polar angle, no wrap needed)."""
    target_theta, target_phi = target
    dtheta = thetas - target_theta
    dphi = (phis - target_phi + np.pi) % (2 * np.pi) - np.pi
    dist = np.sqrt(dtheta**2 + dphi**2)
    order = np.argsort(dist)[:n]
    return [(reps[i], frames[i], thetas[i], phis[i], dist[i]) for i in order]


def write_representative_pdb(state, target, color, hits, out_path):
    """5-MODEL PDB, one MODEL per hit, using the state's own topology."""
    prefix0 = f"{state}/{ff}/{water_model}/{hits[0][0]}"
    template = mda.Universe(f"../workspace/{prefix0}/production_trimmed_static_vacuum.pdb")

    universes = {}
    with mda.Writer(out_path, template.atoms.n_atoms, multiframe=True) as W:
        for replicate_id, frame_idx, theta, phi, dist in hits:
            if replicate_id not in universes:
                prefix = f"{state}/{ff}/{water_model}/{replicate_id}"
                universes[replicate_id] = mda.Universe(
                    f"../workspace/{prefix}/production_trimmed_static_vacuum.pdb",
                    f"../workspace/{prefix}/production_trimmed_static_vacuum.dcd",
                )
            u = universes[replicate_id]
            u.trajectory[frame_idx]
            template.atoms.positions = u.atoms.positions
            W.write(template.atoms)
            print(
                f"    replicate {replicate_id} frame {frame_idx}: "
                f"theta={theta:.3f} phi={phi:.3f} dist={dist:.4f}"
            )


def _tile(thetas, phis):
    theta_tiled = np.concatenate([thetas, thetas, thetas])
    phi_tiled = np.concatenate([phis - 2 * np.pi, phis, phis + 2 * np.pi])
    return theta_tiled, phi_tiled


def hist_max_count(thetas, phis, bins=HIST_BINS):
    """Peak bin count of this state's tiled 2D histogram, for picking a
    colorbar vmax shared across states."""
    theta_tiled, phi_tiled = _tile(thetas, phis)
    h, _, _ = np.histogram2d(theta_tiled, phi_tiled, bins=bins)
    return h.max()


def plot_heatmap_with_circles(state, thetas, phis, out_path, vmax=None):
    theta_tiled, phi_tiled = _tile(thetas, phis)

    fig = plt.figure(figsize=(12, 9))
    plt.hist2d(theta_tiled, phi_tiled, bins=HIST_BINS, cmap="hot", cmin=0, vmax=vmax)
    plt.colorbar(label="Counts")
    plt.title(state)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\phi$")
    plt.xlim(0, np.pi)
    plt.ylim(0, 2 * np.pi)

    ax = plt.gca()
    # Ellipse, not circle: half as long along theta as along phi.
    phi_extent = 0.3
    theta_extent = phi_extent / 2
    for (target_theta, target_phi), color in zip(TARGETS[state], COLORS[state]):
        ellipse = Ellipse(
            (target_theta, target_phi), width=theta_extent, height=phi_extent,
            fill=False, edgecolor=color, linewidth=2.5,
        )
        ax.add_patch(ellipse)

    fig.savefig(out_path)
    plt.close(fig)


def main():
    # Load every state up front (rather than per-state in one pass) so the
    # colorbar vmax can be the max bin count across ALL states' histograms,
    # shared by every plot below -- otherwise each state's plot would
    # saturate at its own peak, making the same count look like a different
    # color between states.
    cv_data = {state: load_theta_phi(state) for state in TARGETS}
    shared_vmax = max(hist_max_count(thetas, phis) for thetas, phis, _, _ in cv_data.values())
    print(f"Shared colorbar vmax across states: {shared_vmax:.0f}")

    for state, (thetas, phis, reps, frames) in cv_data.items():
        print(f"=== {state} ===")

        for (target_theta, target_phi), color in zip(TARGETS[state], COLORS[state]):
            print(f"  target ({target_theta}, {target_phi}) -> {color}")
            hits = nearest_frames(thetas, phis, reps, frames, (target_theta, target_phi))
            out_path = f"{state}_theta{target_theta:.3f}_phi{target_phi:.3f}_{color}.pdb"
            write_representative_pdb(state, (target_theta, target_phi), color, hits, out_path)
            print(f"    wrote {out_path}")

        heatmap_path = f"{state}_CV_heatmap_with_targets.png"
        plot_heatmap_with_circles(state, thetas, phis, heatmap_path, vmax=shared_vmax)
        print(f"  wrote {heatmap_path}")


if __name__ == "__main__":
    main()
