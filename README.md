# PCP1 MD Simulation Pipeline

Molecular dynamics simulations of PCP1 (peptidyl carrier protein 1) across three
functional states, with custom force field parameters for the
4′-phosphopantetheine (PPT) post-translational modification on Ser59.

## System states

| `--state` | Description | PDB file |
|---|---|---|
| `apo` | Unmodified protein (Ser59) | `PCP1_solvated.pdb` |
| `holo` | PPT-serine — pantetheine thiol arm attached | `PCP1_PPT_solvated.pdb` |
| `cys-loaded` | PPT+cysteine — cysteine loaded onto PPT via thioester | `PCP1_cysPPT_solvated.pdb` |

`loaded` is accepted as an alias for `cys-loaded`.

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

### Simulation protocol

| Parameter | Value |
|---|---|
| Integrator | Langevin middle (γ = 1 ps⁻¹) |
| Timestep | 2 fs |
| Temperature | 298.15 K |
| Production length | 1050 ns |
| Non-bonded cutoff | 1.0 nm, PME |
| Constraints | H-bonds |
| Precision | Mixed (CUDA) |

### Output layout

```
workspace/<state>/<ff>/<water>/<seed>/
    system.xml            serialized OpenMM System
    minimized.pdb         energy-minimized structure
    minimized_state.xml   minimized simulation state
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
