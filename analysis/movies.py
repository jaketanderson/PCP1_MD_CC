#!/usr/bin/env python
# coding: utf-8
"""Render side-by-side replicate movies of the production trajectories.

For every condition (state x force field x water model) this produces one mp4
showing all four replicates playing in sync in a 2x2 grid, plus a single 12-up
overview movie showing replicate 0 of every condition.

The pipeline is three stages:

  1. extract  - MDAnalysis strips water/ions from production_trimmed.dcd,
                subsamples every STRIDE frames, repairs periodic wrapping, and
                superposes every frame of every replicate on one global
                reference through the low-RMSF core, so all panels share an
                orientation. Runs in the `openmm2` env (MDAnalysis).
  2. render   - headless PyMOL ray-traces one PNG per frame per replicate. Runs
                as a subprocess out of the `pymol` env, which is the only env
                with PyMOL installed.
  3. encode   - Pillow tiles the panels into a labelled grid, ffmpeg encodes.
                ffmpeg here is built without libfreetype, so all text has to be
                baked into the frames rather than drawn by a filter.

Usage (from the analysis/ directory, under the openmm2 env):

    python movies.py                       # everything, all 12 conditions
    python movies.py --conditions apo/ff14sb/opc --limit-frames 20   # smoke test
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------- constants

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

# Matches analysis.py: production_trimmed_static.dcd already has the 50 ns
# burn-in removed and is aligned on the low-RMSF core, so movie time starts at 0.
REPORT_INTERVAL_PS = 20
BURN_IN_NS = 50

STRIDE = 25          # 25 * 20 ps = 0.5 ns per movie frame
FPS = 30
PANEL_W, PANEL_H = 640, 480

SOLVENT_SEL = "not resname HOH and not resname NA and not resname CL"

# Matches analysis.py: alignment uses the eight lowest-RMSF CA positions.
N_CORE_RESIDUES = 8

# A Ser CB-OG separation above this (Angstrom) means the phosphopantetheine arm
# is not covalently held, so the long CB-OG stick is not a real bond.
PPT_BOND_TOLERANCE = 2.0

# apo keeps two N-terminal residues (GLY THR) that the holo/cys-loaded
# constructs lack, so apo resid N corresponds to holo/cys resid N - 2. The one
# position that still differs in resname is the phosphopantetheine attachment
# site (PPT in holo/cys, SER in apo); it is dropped from the alignment
# selection. Verified against the minimized.pdb topologies.
APO_RESID_OFFSET = 2

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(os.path.dirname(HERE), "workspace")
SCRATCH = os.path.join(HERE, ".movie_frames")
PYMOL_BIN = os.path.expanduser("~/miniconda3/envs/pymol/bin/pymol")
REFERENCE = "apo/ff14sb/opc/0"   # defines the shared camera and orientation

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"


def condition_labels(path):
    """Return readable (state, ff, water_model) labels for a condition or
    replicate path, e.g. 'apo/ff14sb/opc' or 'apo/ff14sb/opc/0'."""
    state, ff, water_model = path.split("/")[:3]
    ff_label = ff[:4] + ff[4:].upper()   # ff14sb -> ff14SB
    water_label = water_model.upper()    # opc -> OPC, tip3p -> TIP3P
    return state, ff_label, water_label


def scratch_dir(replicate):
    return os.path.join(SCRATCH, replicate.replace("/", "_"))


# ---------------------------------------------------------------- stage 1

def _common_ca(universe, state):
    """CA atoms keyed by a state-independent residue number.

    Used only to compute the rigid transform onto the global reference, so
    panels from different states end up in the same orientation. Returns the CA
    AtomGroup and a {shared_resid: index into that group} map; callers
    intersect the maps, which drops any position present in one state only
    (e.g. PPT, which has no CA).
    """
    ca = universe.select_atoms(f"{SOLVENT_SEL} and name CA")
    offset = APO_RESID_OFFSET if state == "apo" else 0
    return ca, {atom.resid - offset: i for i, atom in enumerate(ca)}


def _load(replicate):
    """Universe over the trimmed trajectory, with solute connectivity.

    The trajectory is production_trimmed.dcd rather than the
    production_trimmed_static.dcd analysis.py writes: the latter was rotated by
    the alignment while its stored unit cell was not, so its box record no
    longer describes its coordinates and periodic wrapping cannot be undone in
    it. Movies therefore start from the un-rotated trimmed trajectory and do
    their own alignment.

    Bonds come from distance-guessing over the minimized structure plus
    whatever CONECT records the PDB carries (the PPT residue's are only in
    CONECT); together they connect the solute into a single fragment, which is
    what make_whole needs.
    """
    import MDAnalysis as mda

    topology = f"{WORKSPACE}/{replicate}/minimized.pdb"
    reference = mda.Universe(topology)
    reference_solute = reference.select_atoms(SOLVENT_SEL)
    reference_solute.guess_bonds()

    universe = mda.Universe(topology, f"{WORKSPACE}/{replicate}/production_trimmed.dcd")
    if len(getattr(universe, "bonds", [])):
        universe.delete_bonds(universe.bonds)
    universe.add_bonds(reference_solute.bonds.to_indices())

    solute = universe.select_atoms(SOLVENT_SEL)
    if len(solute.fragments) != 1:
        raise RuntimeError(
            f"{replicate}: solute is {len(solute.fragments)} fragments, "
            "cannot unwrap it as one molecule")
    return universe, solute


def _superpose(mobile, target):
    """Rigid transform (rotation, mobile centroid, target centroid) taking
    `mobile` onto `target` by least-squares superposition."""
    from MDAnalysis.analysis import align

    mobile_com, target_com = mobile.mean(axis=0), target.mean(axis=0)
    rotation, _ = align.rotation_matrix(mobile - mobile_com, target - target_com)
    return rotation, mobile_com, target_com


def _frames(universe, solute, stride, limit_frames):
    """Yield PBC-repaired solute coordinates for the subsampled frames."""
    from MDAnalysis.lib.mdamath import make_whole

    for count, _ in enumerate(universe.trajectory[::stride]):
        if limit_frames and count >= limit_frames:
            return
        make_whole(solute)
        yield solute.positions.copy()


def extract_replicate(replicate, stride, limit_frames, reference):
    """Write a solute-only, subsampled, aligned trajectory for one replicate.

    Each frame is superposed on the global reference through the same low-RMSF
    core residues analysis.py uses, which removes tumbling while leaving the
    flexible regions visibly moving, and leaves every replicate of every
    condition in one shared orientation.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import MDAnalysis as mda

    state = replicate.split("/")[0]
    out = scratch_dir(replicate)
    os.makedirs(out, exist_ok=True)

    universe, solute = _load(replicate)
    ca, resid_map = _common_ca(universe, state)
    ca_in_solute = [list(solute.indices).index(atom.index) for atom in ca]

    core_target = np.asarray(reference["core_positions"], dtype=np.float32)
    core_idx = [ca_in_solute[resid_map[r]] for r in reference["core_resids"]]

    topology_path = os.path.join(out, "solute.pdb")
    traj_path = os.path.join(out, "solute.dcd")

    # The Ser CB-OG bond that anchors the phosphopantetheine arm is missing from
    # the ff14SB systems, so in those runs the arm is a free molecule that
    # drifts off. Tracking the distance lets the render drop the meaningless
    # long stick and the movie say so on the panel.
    linkage = [solute.select_atoms(f"resname PPT and name {name}") for name in ("CB", "OG")]
    linkage_distances = []

    n_written = 0
    with mda.Writer(traj_path, solute.n_atoms) as writer:
        for positions in _frames(universe, solute, stride, limit_frames):
            rotation, mobile_com, target_com = _superpose(positions[core_idx],
                                                          core_target)
            solute.positions = (positions - mobile_com) @ rotation.T + target_com
            if all(group.n_atoms == 1 for group in linkage):
                linkage_distances.append(
                    float(np.linalg.norm(linkage[0].positions[0] - linkage[1].positions[0])))
            if n_written == 0:
                solute.write(topology_path)
            writer.write(solute)
            n_written += 1

    detached = bool(linkage_distances) and float(np.median(linkage_distances)) > PPT_BOND_TOLERANCE
    with open(os.path.join(out, "meta.json"), "w") as handle:
        json.dump({
            "n_frames": n_written,
            "stride": stride,
            "detached_arm": detached,
            "median_cb_og": float(np.median(linkage_distances)) if linkage_distances else None,
        }, handle)
    return n_written


def build_reference(stride, limit_frames):
    """Define the shared alignment target: the reference replicate's first
    frame, plus the residues that fluctuate least across its trajectory.

    Core residues are picked exactly as analysis.py picks them (the
    N_CORE_RESIDUES lowest-RMSF CAs), so the movies show the same frame of
    reference the DCC and RMSF results were computed in. One core set is used
    for every replicate and every state, keyed by the state-independent
    residue numbering, so panels stay comparable.
    """
    import warnings
    warnings.filterwarnings("ignore")

    state = REFERENCE.split("/")[0]
    universe, solute = _load(REFERENCE)
    ca, resid_map = _common_ca(universe, state)
    ca_in_solute = [list(solute.indices).index(atom.index) for atom in ca]
    # Shared resid for each column of the CA stack, in CA-group order.
    resids = [r for r, _ in sorted(resid_map.items(), key=lambda item: item[1])]

    stack = []
    for positions in _frames(universe, solute, stride, limit_frames):
        stack.append(positions[ca_in_solute])
    stack = np.asarray(stack, dtype=np.float32)

    # Fit every frame on the first so the fluctuations measure internal motion
    # rather than tumbling, then take the least mobile residues as the core.
    first = stack[0]
    for i in range(1, len(stack)):
        rotation, mobile_com, target_com = _superpose(stack[i], first)
        stack[i] = (stack[i] - mobile_com) @ rotation.T + target_com
    rmsf = np.sqrt(((stack - stack.mean(axis=0)) ** 2).sum(axis=-1).mean(axis=0))
    core = np.argsort(rmsf)[:N_CORE_RESIDUES]

    return {
        "core_resids": [resids[int(i)] for i in sorted(core)],
        "core_positions": first[sorted(core)].tolist(),
    }


# ---------------------------------------------------------------- stage 2

PML_TEMPLATE = """
load {topology}, mol
load_traj {trajectory}, mol
hide everything
bg_color white
set ray_opaque_background, 1
set ray_shadows, 0
set antialias, 1
set orthoscopic, 1
set cartoon_transparency, 0
show cartoon, polymer
spectrum count, rainbow, polymer and name CA
show sticks, resn PPT
color grey40, resn PPT and elem C
util.cnc("resn PPT and not elem C")
{unbond}
{view}
python
from pymol import cmd
n_states = cmd.count_states("mol")
for state in range(1, n_states + 1):
    cmd.frame(state)
    cmd.dss("mol", state)
    cmd.png("{out}/frame_%05d.png" % state, width={width}, height={height}, ray=1)
python end
"""


def compute_view(reference_replicate):
    """Ray a single frame of the reference with `orient` and capture the camera
    matrix, so every panel of every condition uses one identical view."""
    out = scratch_dir(reference_replicate)
    view_path = os.path.join(SCRATCH, "view.json")
    script = os.path.join(SCRATCH, "view.pml")
    with open(script, "w") as handle:
        handle.write(f"""
load {out}/solute.pdb, ref
orient polymer
# Buffer is generous on purpose: the camera is fixed for the whole run, so the
# flexible termini must stay in frame across every replicate and every state.
zoom polymer, 4
python
import json
from pymol import cmd
json.dump(list(cmd.get_view()), open("{view_path}", "w"))
python end
""")
    subprocess.run([PYMOL_BIN, "-cq", script], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(view_path) as handle:
        return json.load(handle)


def read_meta(replicate):
    with open(os.path.join(scratch_dir(replicate), "meta.json")) as handle:
        return json.load(handle)


def render_replicate(replicate, view):
    """Ray-trace one PNG per frame for a single replicate."""
    out = scratch_dir(replicate)
    panels = os.path.join(out, "panels")
    os.makedirs(panels, exist_ok=True)
    script = os.path.join(out, "render.pml")
    view_cmd = "set_view (" + ", ".join(f"{v:.6f}" for v in view) + ")"
    # Where the arm is not bonded, PyMOL would otherwise draw the CONECT record
    # as a stick stretching across the whole panel.
    unbond = ("unbond resn PPT and name CB, resn PPT and name OG"
              if read_meta(replicate).get("detached_arm") else "")
    with open(script, "w") as handle:
        handle.write(PML_TEMPLATE.format(
            topology=os.path.join(out, "solute.pdb"),
            trajectory=os.path.join(out, "solute.dcd"),
            view=view_cmd,
            unbond=unbond,
            out=panels,
            width=PANEL_W,
            height=PANEL_H,
        ))
    result = subprocess.run([PYMOL_BIN, "-cq", script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pymol failed for {replicate}:\n{result.stderr[-2000:]}")
    return len(os.listdir(panels))


# ---------------------------------------------------------------- stage 3

def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size)


def _panel(path, size, label, font):
    """Load one rendered frame, scale it, and caption it."""
    image = Image.open(path).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), label, fill=(20, 20, 20), font=font)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(180, 180, 180))
    return image


def _encode(frames_dir, out_path, fps):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", os.path.join(frames_dir, "frame_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", out_path],
        check=True,
    )


def _time_ns(frame_index, stride):
    return frame_index * stride * REPORT_INTERVAL_PS / 1000.0


def _panel_label(replicate_id, meta, n_frames):
    label = f"replicate {replicate_id}"
    if meta.get("detached_arm"):
        label += "  [Ppant arm detached]"
    if meta["n_frames"] < n_frames:
        label += f"  [run ended at {_time_ns(meta['n_frames'] - 1, meta['stride']):.0f} ns]"
    return label


def encode_condition(prefix, stride, fps):
    """Tile the four replicates of one condition into a 2x2 grid movie.

    A run that failed or was stopped early (see analysis/README notes on
    holo/ff14sb/opc/2, holo/ff14sb/tip3p/{1,3}, cys-loaded/ff14sb/tip3p/0)
    freezes on its last rendered frame, labelled, rather than truncating the
    whole grid to the shortest replicate and silently hiding the other three
    replicates' extra data.
    """
    replicates = [f"{prefix}/{rid}" for rid in replicate_ids]
    metas = [read_meta(replicate) for replicate in replicates]
    n_frames = max(meta["n_frames"] for meta in metas)

    state, ff_label, water_label = condition_labels(prefix)
    title = f"{state}-PCP1  ({ff_label}, {water_label})"
    header = 44
    grid_w, grid_h = PANEL_W * 2, PANEL_H * 2 + header
    title_font, panel_font = _font(24), _font(20)

    composites = os.path.join(SCRATCH, "composite", prefix.replace("/", "_"))
    shutil.rmtree(composites, ignore_errors=True)
    os.makedirs(composites, exist_ok=True)

    for frame in range(1, n_frames + 1):
        canvas = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), title, fill=(0, 0, 0), font=title_font)
        stamp = f"t = {_time_ns(frame - 1, stride):7.1f} ns  (after {BURN_IN_NS} ns burn-in)"
        draw.text((grid_w - 380, 12), stamp, fill=(60, 60, 60), font=title_font)
        for i, (replicate, meta) in enumerate(zip(replicates, metas)):
            held_frame = min(frame, meta["n_frames"])
            panel = _panel(
                os.path.join(scratch_dir(replicate), "panels", f"frame_{held_frame:05d}.png"),
                (PANEL_W, PANEL_H), _panel_label(replicate_ids[i], meta, n_frames), panel_font,
            )
            canvas.paste(panel, ((i % 2) * PANEL_W, header + (i // 2) * PANEL_H))
        canvas.save(os.path.join(composites, f"frame_{frame:05d}.png"))

    out_dir = os.path.join(HERE, prefix)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "replicates.mp4")
    _encode(composites, out_path, fps)
    shutil.rmtree(composites, ignore_errors=True)
    return out_path


def encode_overview(active_prefixes, stride, fps):
    """One 4x3 grid movie showing replicate 0 of every condition."""
    cell_w, cell_h = PANEL_W // 2, PANEL_H // 2
    cols = 4
    rows = (len(active_prefixes) + cols - 1) // cols
    header = 40
    grid_w, grid_h = cell_w * cols, cell_h * rows + header
    title_font, panel_font = _font(22), _font(14)

    metas = [read_meta(f"{prefix}/0") for prefix in active_prefixes]
    n_frames = max(meta["n_frames"] for meta in metas)

    composites = os.path.join(SCRATCH, "composite", "overview")
    shutil.rmtree(composites, ignore_errors=True)
    os.makedirs(composites, exist_ok=True)

    for frame in range(1, n_frames + 1):
        canvas = Image.new("RGB", (grid_w, grid_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 9), "PCP1 conditions, replicate 0", fill=(0, 0, 0), font=title_font)
        stamp = f"t = {_time_ns(frame - 1, stride):7.1f} ns"
        draw.text((grid_w - 200, 10), stamp, fill=(60, 60, 60), font=title_font)
        for i, (prefix, meta) in enumerate(zip(active_prefixes, metas)):
            state, ff_label, water_label = condition_labels(prefix)
            held_frame = min(frame, meta["n_frames"])
            label = f"{state} {ff_label} {water_label}"
            if meta["n_frames"] < n_frames:
                label += " [ended early]"
            panel = _panel(
                os.path.join(scratch_dir(f"{prefix}/0"), "panels", f"frame_{held_frame:05d}.png"),
                (cell_w, cell_h), label, panel_font,
            )
            canvas.paste(panel, ((i % cols) * cell_w, header + (i // cols) * cell_h))
        canvas.save(os.path.join(composites, f"frame_{frame:05d}.png"))

    out_path = os.path.join(HERE, "overview.mp4")
    _encode(composites, out_path, fps)
    shutil.rmtree(composites, ignore_errors=True)
    return out_path


# ---------------------------------------------------------------- driver

def _extract_task(args):
    replicate, stride, limit_frames, reference = args
    return replicate, extract_replicate(replicate, stride, limit_frames, reference)


def _render_task(args):
    replicate, view = args
    return replicate, render_replicate(replicate, view)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", default=prefixes,
                        help="condition prefixes to process (default: all 12)")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"trajectory frames per movie frame (default {STRIDE})")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--limit-frames", type=int, default=0,
                        help="cap movie frames per replicate (smoke tests)")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("N_WORKERS", 0)) or max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--keep-frames", action="store_true",
                        help="keep the per-frame PNGs in analysis/.movie_frames")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--no-overview", action="store_true")
    args = parser.parse_args()

    unknown = [p for p in args.conditions if p not in prefixes]
    if unknown:
        parser.error(f"unknown condition(s): {unknown}")
    if not os.path.exists(PYMOL_BIN):
        parser.error(f"PyMOL not found at {PYMOL_BIN}")

    replicates = [f"{prefix}/{rid}" for prefix in args.conditions for rid in replicate_ids]
    os.makedirs(SCRATCH, exist_ok=True)
    print(f"{len(args.conditions)} condition(s), {len(replicates)} replicate(s), "
          f"{args.workers} worker(s)")

    if not args.skip_extract:
        reference = build_reference(args.stride, args.limit_frames)
        tasks = [(r, args.stride, args.limit_frames, reference) for r in replicates]
        _run("extract", _extract_task, tasks, args.workers)

    # The camera comes from the global reference when it is in play, otherwise
    # from the first requested condition; either way one view serves every panel
    # because all replicates were transformed onto the same reference frame.
    camera_source = REFERENCE
    if os.path.dirname(REFERENCE) not in args.conditions:
        camera_source = f"{args.conditions[0]}/0"
    view = compute_view(camera_source)

    if not args.skip_render:
        _run("render", _render_task, [(r, view) for r in replicates], args.workers)

    outputs = []
    for prefix in tqdm(args.conditions, desc="encode"):
        outputs.append(encode_condition(prefix, args.stride, args.fps))
    if not args.no_overview and len(args.conditions) > 1:
        outputs.append(encode_overview(args.conditions, args.stride, args.fps))

    if not args.keep_frames:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    print("\nWrote:")
    for path in outputs:
        print(f"  {os.path.relpath(path, HERE)}")


def _run(stage, task, tasks, workers):
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task, item): item[0] for item in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=stage):
            replicate = futures[future]
            try:
                future.result()
            except Exception:
                failures.append(replicate)
                print(f"\n[FAILED {stage}] {replicate}\n{traceback.format_exc()}")
    if failures:
        print(f"{len(failures)} replicate(s) failed during {stage}: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
