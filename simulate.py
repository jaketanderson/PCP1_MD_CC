import argparse
import os
import sys
from pathlib import Path

import openmm
from openmm import app, unit

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
workspace: Path = Path(f"workspace/{state}/{ff}/{water}/{seed}")
workspace.mkdir(parents=True, exist_ok=True)

# ── Simulation parameters ─────────────────────────────────────────────────────
timestep    = 2 * unit.femtosecond
runtime     = (args.runtime_ns if args.runtime_ns is not None else 550) * unit.nanoseconds
temperature = 298.15 * unit.kelvin

# NPT equilibration (production itself is NVT)
equil_time        = args.equil_ns * unit.nanoseconds
pressure          = 1 * unit.bar
barostat_interval = 25   # steps between Monte Carlo volume moves

# ── System setup ─────────────────────────────────────────────────────────────
ff  = app.ForceField(*ff_files)
pdb = app.PDBFile(cfg["pdb"])

modeller = app.Modeller(pdb.topology, pdb.positions)

# OPC water requires a virtual M-site; addExtraParticles inserts any the solute
# needs (no-op for these systems, but harmless and keeps the two models symmetric).
modeller.addExtraParticles(ff)
print(f"Solute: {modeller.topology.getNumAtoms()} atoms")

# The input PDBs are unsolvated, so build the box here. A rhombic dodecahedron
# holds the same padding in ~70% of the volume of a cube. addSolvent also
# neutralizes the net charge (-3 holo, -5 cys-loaded) on top of the background salt.
print(f"Solvating: {solvent_model} geometry, {args.padding} nm padding, "
      f"{args.ionic_strength} M NaCl...")
modeller.addSolvent(
    ff,
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

with open(workspace / "solvated.pdb", "w") as f:
    app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

print("Creating system...")
system = ff.createSystem(
    modeller.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    hydrogenMass=3.0 * unit.amu,
)

with open(workspace / "system.xml", "w") as f:
    f.write(openmm.XmlSerializer.serialize(system))

integrator = openmm.LangevinMiddleIntegrator(temperature, 1 / unit.picosecond, timestep)
integrator.setRandomNumberSeed(seed)
platform = openmm.Platform.getPlatformByName("CUDA")
properties = {"Precision": "mixed"}
simulation = app.Simulation(
    modeller.topology,
    system,
    integrator,
    platform,
    properties,
)

# ── Minimization ─────────────────────────────────────────────────────────────
simulation.context.setPositions(modeller.positions)
print("Minimizing energy...")
simulation.minimizeEnergy(tolerance=0.25 * unit.kilojoules_per_mole / unit.nanometer)
positions = simulation.context.getState(getPositions=True).getPositions()
with open(workspace / "minimized.pdb", "w") as f:
    app.PDBFile.writeFile(simulation.topology, positions, f)
simulation.saveState(str(workspace / "minimized_state.xml"))

# ── NPT equilibration ────────────────────────────────────────────────────────
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
        reportInterval=int(20 * unit.picosecond / timestep),
        step=True, time=True,
        totalEnergy=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, volume=True, density=True, speed=True,
    )
    simulation.reporters.append(equil_reporter)
    simulation.step(int(equil_time / timestep))
    simulation.reporters.remove(equil_reporter)

    state = simulation.context.getState(getPositions=True)
    box = state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    print(f"  Equilibrated box (nm): " + "  ".join(
        f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})" for v in box))

    # The barostat changed the volume, but the Topology still holds the box it was
    # built with; sync it so the CRYST1 record written below is the relaxed box.
    simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with open(workspace / "equilibrated.pdb", "w") as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    simulation.saveState(str(workspace / "equilibrated_state.xml"))

    # Drop the barostat; positions, velocities and the relaxed box are preserved.
    system.removeForce(barostat_idx)
    simulation.context.reinitialize(preserveState=True)

# Restart the clock so production.log starts at step 0 / time 0. Both have to be
# reset: currentStep drives the reporting intervals, but the Time column comes
# from the context, so resetting only the former leaves time offset by the
# equilibration length and out of step with the DCD frame numbering.
simulation.currentStep = 0
simulation.context.setTime(0 * unit.picosecond)

# ── Production ───────────────────────────────────────────────────────────────
print("Simulating production...")
simulation.reporters.append(
    app.StateDataReporter(
        str(workspace / "production.log"),
        reportInterval=int(20 * unit.picosecond / timestep),
        step=True, time=True,
        totalEnergy=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True,
    )
)
simulation.reporters.append(
    app.StateDataReporter(
        sys.stdout,
        reportInterval=int(100 * unit.picosecond / timestep),
        step=True, time=True,
        totalEnergy=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, speed=True,
        totalSteps=int(runtime / timestep),
        progress=True, remainingTime=True,
    )
)
simulation.reporters.append(
    app.DCDReporter(
        str(workspace / "production.dcd"),
        reportInterval=int(20 * unit.picosecond / timestep),
        enforcePeriodicBox=True,
    )
)
simulation.reporters.append(
    app.CheckpointReporter(
        str(workspace / "checkpoint.chk"),
        int(10 * unit.nanoseconds / timestep),
    )
)

simulation.step(int(runtime / timestep))
