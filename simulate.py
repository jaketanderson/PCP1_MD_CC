import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import openmm
from openmm import app, unit

# Exit code that tells HTCondor "I checkpointed cleanly, please run me again".
# Paired with `on_exit_remove = (ExitCode =!= 85)` in the submit file.
CHECKPOINT_EXIT_CODE = 85

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run PCP1 MD simulation")
parser.add_argument(
    "--state",
    required=True,
    choices=["apo", "holo", "cys-loaded", "loaded"],
    help="Protein state: apo | holo | cys-loaded (alias: loaded)",
)
parser.add_argument(
    "--seed",
    required=True,
    type=int,
    help="Random seed (also used as replicate index for output directories)",
)
parser.add_argument(
    "--water",
    required=True,
    type=lambda s: s.lower(),
    choices=["tip3p", "opc"],
    help="Water model: tip3p | opc (case insensitive)",
)
parser.add_argument(
    "--ff",
    required=True,
    type=lambda s: s.lower(),
    choices=["ff14sb", "ff19sb"],
    help="Protein force field: ff14sb | ff19sb (case insensitive)",
)
parser.add_argument(
    "--runtime-ns",
    type=float,
    default=None,
    help="Override simulation runtime in nanoseconds (default: 550 ns)",
)
parser.add_argument(
    "--equil-ns",
    type=float,
    default=1.0,
    help="NPT equilibration length in nanoseconds; 0 disables it (default: 1.0)",
)
parser.add_argument(
    "--workspace-root",
    default="workspace",
    help="Root directory for run output (default: workspace)",
)
parser.add_argument(
    "--padding",
    type=float,
    default=1.0,
    help="Solvent padding around the solute, in nm (default: 1.0)",
)
parser.add_argument(
    "--ionic-strength",
    type=float,
    default=0.15,
    help="Background NaCl concentration in molar (default: 0.15)",
)
parser.add_argument(
    "--checkpoint-ns",
    type=float,
    default=10.0,
    help="Checkpoint interval in nanoseconds (default: 10.0)",
)
parser.add_argument(
    "--max-hours",
    type=float,
    default=6.0,
    help=f"Wall-clock budget per invocation; on expiry the run checkpoints and "
         f"exits {CHECKPOINT_EXIT_CODE} to be requeued. 0 disables (default: 6.0)",
)
parser.add_argument(
    "--restart",
    action="store_true",
    help="Discard any existing checkpoint and start this replicate over",
)
args = parser.parse_args()

state = "cys-loaded" if args.state == "loaded" else args.state
seed  = args.seed
water = args.water
ff    = args.ff

PROTEIN_XML: dict[str, str] = {
    "ff14sb": "amber14-all.xml",
    "ff19sb": "amber19/protein.ff19SB.xml",
}
WATER_XML: dict[str, dict[str, str]] = {
    "ff14sb": {"opc": "amber14/opc.xml",   "tip3p": "amber14/tip3p.xml"},
    "ff19sb": {"opc": "amber19/opc.xml",   "tip3p": "amber19/tip3p.xml"},
}
# Modeller.addSolvent needs the geometry of the water template to place molecules.
# OPC is a 4-site model with TIP4P-family geometry, so it is built from 'tip4pew'
# coordinates; the actual OPC parameters come from the water XML loaded above.
SOLVENT_MODEL: dict[str, str] = {"tip3p": "tip3p", "opc": "tip4pew"}

protein_xml   = PROTEIN_XML[ff]
water_xml     = WATER_XML[ff][water]
solvent_model = SOLVENT_MODEL[water]

print(f"State: {state}  Seed: {seed}  Water: {water}  FF: {ff}")

# ── State → PDB + ForceField files ───────────────────────────────────────────
STATE_CONFIG: dict[str, dict[str, str | None]] = {
    "apo":        {"pdb": "apo.pdb",    "ppt_variant": None},
    "holo":       {"pdb": "holo.pdb",   "ppt_variant": "holo"},
    "cys-loaded": {"pdb": "loaded.pdb", "ppt_variant": "cys"},
}

cfg = STATE_CONFIG[state]
extra_ff: list[str] = ([f"data/{cfg['ppt_variant']}/{ff}/ppt_residue.xml"]
                       if cfg["ppt_variant"] else [])
ff_files: list[str] = [protein_xml, water_xml] + extra_ff

# ── Output directory ──────────────────────────────────────────────────────────
workspace: Path = Path(args.workspace_root) / state / ff / water / str(seed)
workspace.mkdir(parents=True, exist_ok=True)

SOLVATED_PDB = workspace / "solvated.pdb"
SYSTEM_XML   = workspace / "system.xml"
PROD_DCD     = workspace / "production.dcd"
PROD_LOG     = workspace / "production.log"
CHECKPOINT   = workspace / "checkpoint.chk"
CHECKPOINT_META = workspace / "checkpoint.json"

# ── Simulation parameters ─────────────────────────────────────────────────────
timestep    = 2 * unit.femtosecond
runtime     = (args.runtime_ns if args.runtime_ns is not None else 550) * unit.nanoseconds
temperature = 298.15 * unit.kelvin

# NPT equilibration (production itself is NVT)
equil_time        = args.equil_ns * unit.nanoseconds
pressure          = 1 * unit.bar
barostat_interval = 25   # steps between Monte Carlo volume moves

# round(), not int(): these divisions are inexact in floating point, and
# int(550 ns / 2 fs) truncates to 274999999 rather than 275000000. That one lost
# step leaves the total off a frame boundary, so the final frame of every
# replicate is never written, and it knocks the checkpoint interval
# (int -> 4999999) off the reporting interval too.
report_interval     = round(20 * unit.picosecond / timestep)
stdout_interval     = round(100 * unit.picosecond / timestep)
checkpoint_interval = round(args.checkpoint_ns * unit.nanoseconds / timestep)
total_steps         = round(runtime / timestep)

PLATFORM_NAME = "CUDA"
PRECISION     = "mixed"

wall_start = time.monotonic()
deadline   = wall_start + args.max_hours * 3600 if args.max_hours > 0 else None


# ── Trajectory/log rollback helpers ──────────────────────────────────────────
# A checkpoint and a trajectory frame are written at different moments, so after
# an abrupt kill the trajectory is almost always AHEAD of the checkpoint. Rather
# than appending blindly (which duplicates frames and makes time run backwards),
# roll both files back to the checkpoint's step and continue from there. The
# checkpoint is the single source of truth.
DCD_HEADER_BYTES = 276          # from openmm.app.dcdfile.DCDFile.__init__
DCD_NSET_OFFSET  = 8            # where append=True reads the frame count


def dcd_frame_bytes(n_atoms: int, periodic: bool = True) -> int:
    """Bytes per DCD frame: optional box record + three coordinate records."""
    box = 4 + 48 + 4 if periodic else 0
    return box + 3 * (4 + 4 * n_atoms + 4)


def truncate_dcd(path: Path, keep_frames: int, n_atoms: int) -> int:
    """Cut a DCD back to keep_frames and fix its header. Returns frames dropped."""
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    frame_bytes = dcd_frame_bytes(n_atoms)
    size = path.stat().st_size
    payload = size - DCD_HEADER_BYTES
    if payload < 0 or payload % frame_bytes != 0:
        raise RuntimeError(
            f"{path} is not a whole number of frames "
            f"(size={size}, header={DCD_HEADER_BYTES}, frame={frame_bytes}). "
            "Refusing to guess; delete it and use --restart."
        )
    have = payload // frame_bytes
    if have <= keep_frames:
        return 0
    with open(path, "r+b") as fh:
        fh.truncate(DCD_HEADER_BYTES + keep_frames * frame_bytes)
        fh.seek(DCD_NSET_OFFSET)
        fh.write(struct.pack("<i", keep_frames))
    return have - keep_frames


def truncate_log(path: Path, max_step: int) -> int:
    """Drop CSV rows past max_step, keeping the header. Returns rows dropped."""
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    lines = path.read_text().splitlines()
    kept, dropped = [], 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            kept.append(line)
            continue
        try:
            if int(line.split(",")[0]) <= max_step:
                kept.append(line)
            else:
                dropped += 1
        except (ValueError, IndexError):
            kept.append(line)
    if dropped:
        path.write_text("\n".join(kept) + "\n")
    return dropped


def write_checkpoint(simulation) -> None:
    """Checkpoint + sidecar metadata. saveCheckpoint() routes through OpenMM's
    safesave, which writes a temp file and renames it, so a kill mid-write
    cannot corrupt the previous checkpoint."""
    simulation.saveCheckpoint(str(CHECKPOINT))
    meta = {
        "openmm_version": openmm.__version__,
        "platform": PLATFORM_NAME,
        "precision": PRECISION,
        "n_particles": simulation.system.getNumParticles(),
        "step": simulation.currentStep,
        "total_steps": total_steps,
        "timestep_fs": timestep.value_in_unit(unit.femtosecond),
        "state": state, "ff": ff, "water": water, "seed": seed,
    }
    tmp = CHECKPOINT_META.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, CHECKPOINT_META)


# ── Decide: resume or start fresh ────────────────────────────────────────────
resume = CHECKPOINT.is_file() and CHECKPOINT_META.is_file() and not args.restart

if args.restart:
    for p in (CHECKPOINT, CHECKPOINT_META, PROD_DCD, PROD_LOG):
        p.unlink(missing_ok=True)
    print("--restart: discarded existing checkpoint and production output")

if resume:
    # ── Resume ───────────────────────────────────────────────────────────────
    # Critically, do NOT re-solvate: Modeller.addSolvent places ions with
    # random.choice, so a rebuild would produce a different system and the
    # checkpoint would not load. Reuse the exact topology and System on disk.
    meta = json.loads(CHECKPOINT_META.read_text())
    print(f"Resuming from {CHECKPOINT} (step {meta.get('step')} of {total_steps})")

    for missing in [p for p in (SOLVATED_PDB, SYSTEM_XML) if not p.is_file()]:
        sys.exit(f"FATAL: resume needs {missing}, which is absent. Use --restart.")

    # saveCheckpoint() is only valid for an identical System, Platform and OpenMM
    # version. Fail loudly rather than let loadCheckpoint die cryptically.
    for key, actual in (("openmm_version", openmm.__version__),
                        ("platform", PLATFORM_NAME),
                        ("precision", PRECISION)):
        if key in meta and meta[key] != actual:
            sys.exit(
                f"FATAL: checkpoint was written with {key}={meta[key]!r} but this "
                f"run uses {actual!r}. Binary checkpoints are not portable across "
                f"that change. Re-run with --restart to start this replicate over."
            )

    pdb = app.PDBFile(str(SOLVATED_PDB))
    topology = pdb.topology
    system = openmm.XmlSerializer.deserialize(SYSTEM_XML.read_text())

    if "n_particles" in meta and meta["n_particles"] != system.getNumParticles():
        sys.exit(f"FATAL: checkpoint has {meta['n_particles']} particles, "
                 f"{SYSTEM_XML} has {system.getNumParticles()}. Use --restart.")

    integrator = openmm.LangevinMiddleIntegrator(
        temperature, 1 / unit.picosecond, timestep)
    integrator.setRandomNumberSeed(seed)
    simulation = app.Simulation(
        topology, system, integrator,
        openmm.Platform.getPlatformByName(PLATFORM_NAME),
        {"Precision": PRECISION},
    )
    simulation.loadCheckpoint(str(CHECKPOINT))

    # The checkpoint's own clock is authoritative for how far production got.
    # Deriving the step from it (rather than trusting the sidecar) keeps the
    # rollback correct even if the sidecar write was interrupted.
    restored = simulation.context.getState()
    steps_done = int(round(restored.getTime() / timestep))
    simulation.currentStep = steps_done
    simulation.topology.setPeriodicBoxVectors(restored.getPeriodicBoxVectors())

    expected_frames = steps_done // report_interval
    n_atoms = system.getNumParticles()
    dropped_frames = truncate_dcd(PROD_DCD, expected_frames, n_atoms)
    dropped_rows = truncate_log(PROD_LOG, steps_done)
    print(f"  Restored to step {steps_done} "
          f"({restored.getTime().value_in_unit(unit.nanoseconds):.3f} ns)")
    print(f"  Rolled trajectory back to {expected_frames} frames "
          f"(dropped {dropped_frames}); dropped {dropped_rows} log rows")

    if steps_done >= total_steps:
        print("Production already complete; nothing to do.")
        sys.exit(0)

else:
    # ── Fresh start ──────────────────────────────────────────────────────────
    # Clear any partial output from an attempt that died before its first
    # checkpoint, so the appending reporters below cannot splice onto it.
    for p in (PROD_DCD, PROD_LOG, CHECKPOINT, CHECKPOINT_META):
        p.unlink(missing_ok=True)

    forcefield = app.ForceField(*ff_files)
    pdb = app.PDBFile(cfg["pdb"])

    modeller = app.Modeller(pdb.topology, pdb.positions)

    # OPC water requires a virtual M-site; addExtraParticles inserts any the solute
    # needs (no-op for these systems, but harmless and keeps the two models symmetric).
    modeller.addExtraParticles(forcefield)
    print(f"Solute: {modeller.topology.getNumAtoms()} atoms")

    # The input PDBs are unsolvated, so build the box here. A rhombic dodecahedron
    # holds the same padding in ~70% of the volume of a cube. addSolvent also
    # neutralizes the net charge (-3 holo, -5 cys-loaded) on top of the background salt.
    print(f"Solvating: {solvent_model} geometry, {args.padding} nm padding, "
          f"{args.ionic_strength} M NaCl...")
    modeller.addSolvent(
        forcefield,
        model=solvent_model,
        boxShape="dodecahedron",
        padding=args.padding * unit.nanometer,
        ionicStrength=args.ionic_strength * unit.molar,
        positiveIon="Na+",
        negativeIon="Cl-",
        neutralize=True,
    )

    box = modeller.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    n_wat = sum(1 for r in modeller.topology.residues() if r.name in ("HOH", "WAT"))
    n_ion = sum(1 for r in modeller.topology.residues() if r.name in ("NA", "CL"))
    print(f"Topology: {modeller.topology.getNumAtoms()} atoms, {n_wat} waters, {n_ion} ions")
    # Triclinic cell: report the vectors themselves, not just their diagonal, since
    # the dodecahedron's third vector is (a/2, a/2, a/sqrt(2)).
    print("Box vectors (nm): " + "  ".join(
        f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})" for v in box))

    with open(SOLVATED_PDB, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    print("Creating system...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=3.0 * unit.amu,
    )

    with open(SYSTEM_XML, "w") as f:
        f.write(openmm.XmlSerializer.serialize(system))

    integrator = openmm.LangevinMiddleIntegrator(
        temperature, 1 / unit.picosecond, timestep)
    integrator.setRandomNumberSeed(seed)
    simulation = app.Simulation(
        modeller.topology, system, integrator,
        openmm.Platform.getPlatformByName(PLATFORM_NAME),
        {"Precision": PRECISION},
    )

    # ── Minimization ─────────────────────────────────────────────────────────
    simulation.context.setPositions(modeller.positions)
    print("Minimizing energy...")
    simulation.minimizeEnergy(tolerance=0.25 * unit.kilojoules_per_mole / unit.nanometer)
    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(workspace / "minimized.pdb", "w") as f:
        app.PDBFile.writeFile(simulation.topology, positions, f)
    simulation.saveState(str(workspace / "minimized_state.xml"))

    # ── NPT equilibration ────────────────────────────────────────────────────
    # The box comes straight from addSolvent, so its density is only approximate.
    # Relax the volume under a barostat before production, then take the barostat
    # back out so production runs at fixed volume (NVT), as before.
    simulation.context.setVelocitiesToTemperature(temperature, seed)

    if equil_time > 0 * unit.nanoseconds:
        print(f"Equilibrating (NPT, {equil_time.value_in_unit(unit.nanoseconds)} ns)...")
        barostat_idx = system.addForce(
            openmm.MonteCarloBarostat(pressure, temperature, barostat_interval)
        )
        simulation.context.reinitialize(preserveState=True)

        equil_reporter = app.StateDataReporter(
            str(workspace / "equilibration.log"),
            reportInterval=report_interval,
            step=True, time=True,
            totalEnergy=True, potentialEnergy=True, kineticEnergy=True,
            temperature=True, volume=True, density=True, speed=True,
        )
        simulation.reporters.append(equil_reporter)
        simulation.step(round(equil_time / timestep))
        simulation.reporters.remove(equil_reporter)

        eq_state = simulation.context.getState(getPositions=True)
        box = eq_state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
        print("  Equilibrated box (nm): " + "  ".join(
            f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})" for v in box))

        # The barostat changed the volume, but the Topology still holds the box it was
        # built with; sync it so the CRYST1 record written below is the relaxed box.
        simulation.topology.setPeriodicBoxVectors(eq_state.getPeriodicBoxVectors())
        with open(workspace / "equilibrated.pdb", "w") as f:
            app.PDBFile.writeFile(simulation.topology, eq_state.getPositions(), f)
        simulation.saveState(str(workspace / "equilibrated_state.xml"))

        # Drop the barostat; positions, velocities and the relaxed box are preserved.
        system.removeForce(barostat_idx)
        simulation.context.reinitialize(preserveState=True)

    # Restart the clock so production.log starts at step 0 / time 0. Both have to be
    # reset: currentStep drives the reporting intervals, but the Time column comes
    # from the context, so resetting only the former leaves time offset by the
    # equilibration length and out of step with the DCD frame numbering. It also
    # makes the context clock a faithful record of production progress, which the
    # resume path above relies on.
    simulation.currentStep = 0
    simulation.context.setTime(0 * unit.picosecond)
    steps_done = 0

# ── Production ───────────────────────────────────────────────────────────────
# append=resume so a requeued job continues the same trajectory and log rather
# than starting new ones.
simulation.reporters.append(
    app.StateDataReporter(
        str(PROD_LOG),
        reportInterval=report_interval,
        step=True, time=True,
        totalEnergy=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True,
        append=resume,
    )
)
simulation.reporters.append(
    app.StateDataReporter(
        sys.stdout,
        reportInterval=stdout_interval,
        step=True, time=True,
        totalEnergy=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True,
        totalSteps=total_steps,
        progress=True, remainingTime=True,
    )
)
simulation.reporters.append(
    app.DCDReporter(
        str(PROD_DCD),
        reportInterval=report_interval,
        append=resume,
        enforcePeriodicBox=True,
    )
)

remaining_ns = (total_steps - steps_done) * timestep.value_in_unit(unit.nanoseconds)
print(f"Simulating production: {remaining_ns:.2f} ns remaining "
      f"(step {steps_done} -> {total_steps})")
if deadline is not None:
    print(f"  Wall-clock budget {args.max_hours} h; will checkpoint and exit "
          f"{CHECKPOINT_EXIT_CODE} if it expires before completion")

# Step in checkpoint-sized chunks so progress is never more than one interval
# ahead of the last durable state.
# Step in reporting-interval slices and test the wall clock after each one.
# Checking only at checkpoint boundaries would let the budget overshoot by a
# whole chunk -- with the default 10 ns interval that is 5,000,000 steps, tens
# of minutes on an L40S, which both risks the 8 h limit and makes a short
# --max-hours (as used to test requeuing) look like it does nothing at all.
#
# Slicing on report_interval also means a deadline checkpoint lands on a frame
# boundary, so the rollback on resume discards nothing.
deadline_check_interval = min(report_interval, checkpoint_interval)
next_checkpoint = ((steps_done // checkpoint_interval) + 1) * checkpoint_interval

while simulation.currentStep < total_steps:
    target = min(simulation.currentStep + deadline_check_interval, total_steps)
    simulation.step(target - simulation.currentStep)

    finished    = simulation.currentStep >= total_steps
    out_of_time = deadline is not None and time.monotonic() >= deadline

    if finished or out_of_time or simulation.currentStep >= next_checkpoint:
        write_checkpoint(simulation)
        while next_checkpoint <= simulation.currentStep:
            next_checkpoint += checkpoint_interval

    if finished:
        break
    if out_of_time:
        done_ns = simulation.currentStep * timestep.value_in_unit(unit.nanoseconds)
        elapsed_h = (time.monotonic() - wall_start) / 3600
        print(f"\nWall-clock budget reached after {elapsed_h:.2f} h at "
              f"{done_ns:.3f} ns (step {simulation.currentStep} of {total_steps}).")
        print(f"Checkpointed; exiting {CHECKPOINT_EXIT_CODE} to be requeued.")
        sys.exit(CHECKPOINT_EXIT_CODE)

elapsed_h = (time.monotonic() - wall_start) / 3600
print(f"\nProduction complete: {total_steps} steps "
      f"({runtime.value_in_unit(unit.nanoseconds):g} ns), "
      f"{elapsed_h:.2f} h this invocation.")
