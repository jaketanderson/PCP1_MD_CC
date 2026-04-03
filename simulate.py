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
protein_xml = PROTEIN_XML[ff]
water_xml   = WATER_XML[ff][water]

print(f"State: {state}  Seed: {seed}  Water: {water}  FF: {ff}")

# ── State → PDB + ForceField files ───────────────────────────────────────────
STATE_CONFIG: dict[str, dict[str, str | None]] = {
    "apo":        {"pdb": "PCP1_solvated.pdb",        "ppt_variant": None},
    "holo":       {"pdb": "PCP1_PPT_solvated.pdb",    "ppt_variant": "holo"},
    "cys-loaded": {"pdb": "PCP1_cysPPT_solvated.pdb", "ppt_variant": "cys"},
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
runtime     = 20 * unit.nanoseconds
temperature = 298.15 * unit.kelvin

# ── System setup ─────────────────────────────────────────────────────────────
ff  = app.ForceField(*ff_files)
pdb = app.PDBFile(cfg["pdb"])

# OPC water requires a virtual M-site; addExtraParticles inserts it (no-op for TIP3P)
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.addExtraParticles(ff)
print(f"Topology: {modeller.topology.getNumAtoms()} atoms")

print("Creating system...")
system = ff.createSystem(
    modeller.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
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
simulation.minimizeEnergy(tolerance=0.5 * unit.kilojoules_per_mole / unit.nanometer)
positions = simulation.context.getState(getPositions=True).getPositions()
with open(workspace / "minimized.pdb", "w") as f:
    app.PDBFile.writeFile(simulation.topology, positions, f)
simulation.saveState(str(workspace / "minimized_state.xml"))

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
