# PCP1 MD Simulation Pipeline

Molecular dynamics simulations of PCP1 (peptidyl carrier protein 1) across three
functional states, with custom force field parameters for the
4′-phosphopantetheine (PPT) post-translational modification on Ser59.

## System states

| `--state` | Description | PDB file | Solute charge |
|---|---|---|---|
| `apo` | Unmodified protein (Ser59) | `apo.pdb` | −2 |
| `holo` | PPT-serine — pantetheine thiol arm attached | `holo.pdb` | −3 |
| `cys-loaded` | PPT+cysteine — cysteine loaded onto PPT via thioester | `loaded.pdb` | −5 |

`loaded` is accepted as an alias for `cys-loaded`.

All three are the same 82-residue construct (ALA22–NME103) and differ only at
position 59 (`SER` → `PPT`). They are **not** solvated — `simulate.py` builds the
water box itself (see below).

## Supported force fields and water models

| `--ff` | Protein FF | Notes |
|---|---|---|
| `ff14sb` | AMBER ff14SB (`amber14-all.xml`) | CA type `protein-CX` |
| `ff19sb` | AMBER ff19SB (`amber19/protein.ff19SB.xml`) | CA type `protein-XC`; canonical residues get residue-specific CMAPs; PTM residue does not |

| `--water` | Model |
|---|---|
| `tip3p` | TIP3P (3-point) |
| `opc` | OPC (4-point, virtual M-site) |

All arguments are case-insensitive.

---

## Environment

Both scripts require the `openmm2` conda environment:

```bash
conda activate openmm2
# or prefix every command with:
conda run -n openmm2 python ...
```

Key packages: OpenMM, OpenFF Toolkit + Interchange, RDKit, AmberTools (for AM1-BCC).

---

## Step 1 — Parameterize the PTM residue

`parameterize.py` builds a capped ACE–PPTSer–NME trimer, computes AM1-BCC partial
charges with OpenFF Sage 2.3.0, extracts bonded and non-bonded parameters, and
writes a custom OpenMM `ForceField` XML for the PPT residue.

**Charges are expensive** (several minutes per variant via `sqm`). They are cached
in `data/<variant>/trimer_interchange.json` and never recomputed unless you
explicitly omit `--skip-charges`.

### First run (computes charges)

```bash
python parameterize.py                          # all variants, all FFs
python parameterize.py --variant holo           # one variant, all FFs
python parameterize.py --variant cys --ff ff19sb
```

### Subsequent runs (reuse cached charges)

```bash
python parameterize.py --skip-charges           # regenerate all XMLs, no recharging
python parameterize.py --variant holo --ff ff14sb --skip-charges
```

### Arguments

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--variant` | `holo`, `cys`, `all` | `all` | Which PTM variant to parameterize |
| `--ff` | `ff14sb`, `ff19sb`, `all` | `all` | Which protein FF to generate the XML for |
| `--skip-charges` | — | off | Load `trimer_interchange.json` instead of rerunning AM1-BCC |

### Output layout

```
data/
  <variant>/                        # FF-independent (written once per variant)
    trimer_interchange.json         ← AM1-BCC charges cached here
    trimer_charged.sdf
    atom_index_to_name.json
    h_names_by_parent.json

  <variant>/<ff>/                   # FF-specific
    parameters.json
    ppt_residue.xml                 ← loaded by simulate.py
```

The `apo` state uses only standard AMBER atom types and needs no parameterization.

---

## Step 2 — Run the simulation

```bash
python simulate.py --state holo --ff ff19sb --water opc --seed 1
```

### Arguments

| Argument | Choices | Required | Description |
|---|---|---|---|
| `--state` | `apo`, `holo`, `cys-loaded`, `loaded` | yes | Protein state |
| `--ff` | `ff14sb`, `ff19sb` | yes | Protein force field |
| `--water` | `tip3p`, `opc` | yes | Water model |
| `--seed` | integer | yes | Random seed; also used as the replicate index in the output path |
| `--runtime-ns` | float | no | Production length override (default 550 ns) |
| `--equil-ns` | float | no | NPT equilibration length; `0` disables (default 1.0 ns) |
| `--padding` | float | no | Solvent padding in nm (default 1.0) |
| `--ionic-strength` | float | no | Background NaCl in molar (default 0.15) |

### Solvation

The input PDBs are dry, so `simulate.py` calls `Modeller.addSolvent` with a
**rhombic dodecahedral** box (~70% the volume of a cube at equal padding),
1.0 nm padding and 0.15 M NaCl, neutralizing the solute charge on top of the
background salt. At the defaults every state lands on a 6.17 nm box with
~4.5k waters (≈14.7k atoms with TIP3P, ≈19.3k with OPC).

OPC is a 4-point model, so `addSolvent` is given `model='tip4pew'` for the water
*geometry* while the OPC *parameters* come from the water XML — the M-site
virtual sites are then created by `createSystem` (one per water).

### Simulation protocol

Minimize → NPT equilibration → NVT production.

| Parameter | Value |
|---|---|
| Integrator | Langevin middle (γ = 1 ps⁻¹) |
| Timestep | 2 fs (H mass 3 amu) |
| Temperature | 298.15 K |
| Equilibration | 1 ns NPT, Monte Carlo barostat at 1 bar (every 25 steps) |
| Production length | 550 ns NVT |
| Non-bonded cutoff | 1.0 nm, PME |
| Constraints | H-bonds |
| Precision | Mixed (CUDA) |

Because `addSolvent` only approximates the density, the box is relaxed under a
barostat before production. Velocities are seeded from the Maxwell–Boltzmann
distribution at 298.15 K using `--seed`, so replicates differ in their starting
velocities as well as in the thermostat's random stream. The barostat is then
removed and the context reinitialized, so production runs at the *equilibrated*
fixed volume. The step counter is reset afterwards, so `production.log` starts
at step 0.

`system.xml` is serialized before the barostat is added and the barostat is
removed again afterwards, so the file on disk matches the system used in
production.

### Output layout

```
workspace/<state>/<ff>/<water>/<seed>/
    solvated.pdb            solvated/neutralized starting structure
    system.xml              serialized OpenMM System
    minimized.pdb           energy-minimized structure
    minimized_state.xml     minimized simulation state
    equilibrated.pdb        post-NPT structure (CRYST1 = relaxed box)
    equilibrated_state.xml  post-NPT simulation state
    equilibration.log       NPT energies / temperature / volume / density
    production.dcd        trajectory (written every 20 ps)
    production.log        energies / temperature / speed (every 20 ps)
    checkpoint.chk        checkpoint (every 10 ns)
```

The `<ff>` level in the path ensures ff14sb and ff19sb runs with the same seed
never collide.

---

## Typical workflow

```bash
# 1. Parameterize (once; charges cached for future use)
python parameterize.py

# 2. Run replicates — mix and match states, FFs, and water models
python simulate.py --state apo        --ff ff19sb --water opc   --seed 1
python simulate.py --state holo       --ff ff19sb --water opc   --seed 1
python simulate.py --state cys-loaded --ff ff19sb --water opc   --seed 1

# Re-run with a different FF without recharging
python parameterize.py --ff ff14sb --skip-charges
python simulate.py --state holo --ff ff14sb --water tip3p --seed 1
```

---

## Force field details for the PPT residue

The PPT modification is treated as a single non-standard residue (`PPT`) that
replaces the modified serine in the PDB topology.

- **Backbone and Cβ atoms** (N, H, CA, HA, C, O, CB, HB2, HB3) borrow types
  directly from the chosen protein FF so that cross-terms at the Ser backbone
  are handled consistently. No parameters for these atom pairs are written into
  `ppt_residue.xml`.
- **PPT sidechain atoms** receive new `PPT_*` types parameterized by OpenFF Sage
  2.3.0. Cross-terms at the Cβ–OG boundary (where the PPT arm attaches) come
  from Sage.
- **Partial charges** are AM1-BCC, fitted on the full capped trimer (ACE–PPTSer–NME)
  and renormalized to the exact formal charge (−1 e for both variants).
- **ff19SB and CMAPs**: the PTM residue uses plain `protein-N`/`protein-XC`/`protein-C`
  type names (not the `cmap-SER-*` residue-specific names), so OpenMM assigns no
  CMAP to it while all canonical residues continue to receive their CMAPs.

### Two failure modes to keep in mind

Both were live in this repo and both fail **silently** — OpenMM reports no error
and the simulation runs to completion with wrong physics. `parameterize.py` also
self-reported success in both cases, because it validated the numbers it wrote
into the XML rather than what OpenMM assigns from it.

1. **Atom *types* and *classes* are not interchangeable.** `amber14/protein.ff14SB.xml`
   declares `<Type class="2C" name="protein-2C"/>` — bare AMBER classes — while
   `amber19/protein.ff19SB.xml` declares `<Type class="protein-2C" name="protein-2C"/>`.
   Emitting `class="protein-2C"` into an ff14sb XML matches nothing, and OpenMM
   drops unmatched bonded terms without warning. This removed the CB–OG bond plus
   3 angles and 10 torsions at the PPT attachment point under ff14sb only.
   `PROTEIN_FF_TYPES` therefore stores `(type, class, mass)` per FF, and "is this
   atom borrowed from the protein FF?" is tested against `PROTEIN_TYPE_NAMES`,
   never against class names.

2. **Never write `charge=` on `<Atom>` inside `<NonbondedForce>`.** An explicit
   per-type charge takes precedence over the `<Residue>` template charge, so
   `charge="0"` zeroes the atom. AMBER's own XMLs omit the attribute and declare
   `<UseAttributeFromResidue name="charge"/>` instead; the generated
   `ppt_residue.xml` must do the same. Previously every `PPT_*` atom was assigned
   charge 0 — the phosphate, both amides, the hydroxyl and the thiol were all
   electrostatically invisible, leaving the PPT residue at +0.18 e instead of −1.

To re-check both after any change, confirm the built system has an integer net
charge, a PPT residue summing to exactly −1, and a CB–OG term in the
`HarmonicBondForce` — under *both* force fields.
