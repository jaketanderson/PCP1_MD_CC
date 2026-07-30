#!/usr/bin/env python
"""
parameterize.py  –  Generate OpenMM ForceField XML for PPT-modified serines in PCP1
=====================================================================================
Builds an ACE-PPTSer-NME capped trimer for each variant, parameterizes with OpenFF
Sage 2.3.0 (openff-2.3.0.offxml) and AM1-BCC charges, extracts per-residue
parameters, and writes a custom OpenMM ForceField XML.

Variants
--------
  holo       PPT-serine: phosphopantetheine thiol (PCP1_PPT_solvated.pdb)
               27 PPT heavy atoms, ends …CP8–S01(SH)
  cys        CysPPT-serine: phosphopantetheine + cysteine via amide
               (PCP1_cysPPT_solvated.pdb)
               33 PPT heavy atoms, ends …CP8–NR8(H)–CS8(=OT8)–CT9–CY10–SF10

Both variants define a residue named PPT in their respective XML files.

Backbone/CB atoms borrow types from the chosen protein FF (ff14sb or ff19sb);
all PPT sidechain atoms get new PPT_* types.  Cross-term parameters at the
CB–OG boundary come from Sage.  AM1-BCC charges are FF-independent and cached
in trimer_interchange.json; the XML is regenerated per-FF without recharging.

Usage
-----
    conda run -n openmm2 python parameterize.py                  # all variants × all FFs
    conda run -n openmm2 python parameterize.py --variant holo --ff ff19sb
    conda run -n openmm2 python parameterize.py --skip-charges   # reuse cached charges

Outputs
--------
    data/<variant>/                        (FF-independent, written once)
        atom_index_to_name.json
        h_names_by_parent.json
        trimer_charged.sdf
        trimer_interchange.json            ← AM1-BCC charges cached here

    data/<variant>/<ff>/                   (FF-specific)
        parameters.json
        ppt_residue.xml
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from xml.dom import minidom

import openmm
from openmm import unit as mm_unit
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, ForceField as OFFForceField, Topology
from openff.interchange import Interchange

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")

# ============================================================================
# SHARED CONSTANTS
# ============================================================================

_S = Chem.BondType.SINGLE
_D = Chem.BondType.DOUBLE

# Backbone + CB atom types for each supported protein FF, as (type, class, mass).
# Only CA differs: ff14SB uses protein-CX; ff19SB uses protein-XC.
# Using plain (non-cmap-) type names for the PTM residue means it gets no CMAP
# while all canonical residues still get their residue-specific CMAPs.
#
# NB: type and class names are NOT interchangeable.  amber14/protein.ff14SB.xml
# declares <Type class="2C" name="protein-2C"/> — bare AMBER classes — whereas
# amber19/protein.ff19SB.xml declares <Type class="protein-2C" name="protein-2C"/>.
# Writing class="protein-2C" into an ff14sb XML matches nothing, and OpenMM drops
# unmatched bonded terms *silently* (this previously dropped the CB–OG bond and
# every other cross-term at the PPT attachment point under ff14sb).
PROTEIN_FF_TYPES: dict[str, dict[str, tuple[str, str, float]]] = {
    "ff14sb": {
        "N":   ("protein-N",  "N",  14.007),
        "H01": ("protein-H",  "H",   1.008),
        "CA":  ("protein-CX", "CX", 12.011),
        "HA":  ("protein-H1", "H1",  1.008),
        "C":   ("protein-C",  "C",  12.011),
        "O":   ("protein-O",  "O",  15.999),
        "CB":  ("protein-2C", "2C", 12.011),
        "HB2": ("protein-H1", "H1",  1.008),
        "HB3": ("protein-H1", "H1",  1.008),
    },
    "ff19sb": {
        "N":   ("protein-N",  "protein-N",  14.007),
        "H01": ("protein-H",  "protein-H",   1.008),
        "CA":  ("protein-XC", "protein-XC", 12.011),
        "HA":  ("protein-H1", "protein-H1",  1.008),
        "C":   ("protein-C",  "protein-C",  12.011),
        "O":   ("protein-O",  "protein-O",  15.999),
        "CB":  ("protein-2C", "protein-2C", 12.011),
        "HB2": ("protein-H1", "protein-H1",  1.008),
        "HB3": ("protein-H1", "protein-H1",  1.008),
    },
}
PROTEIN_CLASS_NAMES: dict[str, set[str]] = {ff: {v[1] for v in types.values()} for ff, types in PROTEIN_FF_TYPES.items()}
PROTEIN_TYPE_NAMES:  dict[str, set[str]] = {ff: {v[0] for v in types.values()} for ff, types in PROTEIN_FF_TYPES.items()}

_ELEMENT_MASS: dict[str, float] = {"C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06, "H": 1.008}


# ============================================================================
# VARIANT CONFIGURATIONS
# ============================================================================
# Each config dict holds:
#   heavy_atoms         : list of (pdb_name, atomic_num, formal_charge)
#                         indices 0..n_ppt_heavy-1 = PPT residue, remainder = caps
#   bonds               : list of (idx_a, idx_b, BondType)
#   h_names_by_parent   : dict {heavy_pdb_name: [h_names_in_addHs_order]}
#   residue_atom_order  : ordered list of all PPT residue atom names (heavy + H)
#   n_ppt_heavy         : number of heavy atoms belonging to the PPT residue
#   expected_total_atoms: total atoms after AddHs (for assertion)
#   formal_charge       : expected net formal charge of the trimer

# ─────────────────────────────────────────────────────────────────────────────
# HOLO: phosphopantetheine thiol  (27 PPT heavy atoms + 5 caps = 32 heavy)
# ─────────────────────────────────────────────────────────────────────────────
HOLO_HEAVY_ATOMS = [
    # PPT residue heavy atoms (indices 0–26)
    ("N",       7,  0),   # 0  backbone NH
    ("CA",      6,  0),   # 1  alpha-carbon
    ("C",       6,  0),   # 2  backbone carbonyl C
    ("O",       8,  0),   # 3  backbone carbonyl O
    ("CB",      6,  0),   # 4  beta-carbon
    ("OG",      8,  0),   # 5  serine Oγ → bridge to P
    ("OE1",     8,  0),   # 6  phosphoryl =O
    ("OE2",     8,  0),   # 7  P→pantothenol ester O
    ("OE3",     8, -1),   # 8  phosphate [O⁻]
    ("CZ2",     6,  0),   # 9  pantothenol OCH₂
    ("CH2",     6,  0),   # 10 quaternary C (0 H despite name)
    ("CI6",     6,  0),   # 11 pantoate–β-ala amide C=O
    ("OI2",     8,  0),   # 12 pantothenate OH
    ("NK7",     7,  0),   # 13 pantoate–β-ala amide N
    ("OK6",     8,  0),   # 14 pantoate–β-ala amide =O
    ("CL7",     6,  0),   # 15 β-ala CH₂ (α to N)
    ("CM7",     6,  0),   # 16 β-ala CH₂ (α to amide C)
    ("CN7",     6,  0),   # 17 β-ala–cysteamine amide C=O
    ("PD",     15,  0),   # 18 phosphorus
    ("NX8",     7,  0),   # 19 β-ala–cysteamine amide N
    ("OX7",     8,  0),   # 20 β-ala–cysteamine amide =O
    ("CO8",     6,  0),   # 21 cysteamine CH₂ (α to N)
    ("CP8",     6,  0),   # 22 cysteamine CH₂ (α to S)
    ("CQ2",     6,  0),   # 23 pantoate chiral CH
    ("CQ4",     6,  0),   # 24 gem-methyl 1
    ("CQ5",     6,  0),   # 25 gem-methyl 2
    ("S01",    16,  0),   # 26 thiol S
    # ACE/NME caps (indices 27–31)
    ("ACE_C",   6,  0),   # 27
    ("ACE_O",   8,  0),   # 28
    ("ACE_CH3", 6,  0),   # 29
    ("NME_N",   7,  0),   # 30
    ("NME_CH3", 6,  0),   # 31
]

HOLO_BONDS = [
    (0,  1,  _S),   # N–CA
    (1,  2,  _S),   # CA–C
    (2,  3,  _D),   # C=O
    (1,  4,  _S),   # CA–CB
    (4,  5,  _S),   # CB–OG
    (5,  18, _S),   # OG–PD
    (18, 6,  _D),   # PD=OE1
    (18, 7,  _S),   # PD–OE2
    (18, 8,  _S),   # PD–OE3
    (7,  9,  _S),   # OE2–CZ2
    (9,  10, _S),   # CZ2–CH2
    (10, 23, _S),   # CH2–CQ2
    (10, 24, _S),   # CH2–CQ4
    (10, 25, _S),   # CH2–CQ5
    (23, 11, _S),   # CQ2–CI6
    (23, 12, _S),   # CQ2–OI2
    (11, 14, _D),   # CI6=OK6
    (11, 13, _S),   # CI6–NK7
    (13, 15, _S),   # NK7–CL7
    (15, 16, _S),   # CL7–CM7
    (16, 17, _S),   # CM7–CN7
    (17, 20, _D),   # CN7=OX7
    (17, 19, _S),   # CN7–NX8
    (19, 21, _S),   # NX8–CO8
    (21, 22, _S),   # CO8–CP8
    (22, 26, _S),   # CP8–S01
    # ACE cap
    (27, 28, _D),   # ACE_C=ACE_O
    (27, 29, _S),   # ACE_C–ACE_CH3
    (27, 0,  _S),   # ACE_C–N
    # NME cap
    (2,  30, _S),   # C–NME_N
    (30, 31, _S),   # NME_N–NME_CH3
]

HOLO_H_NAMES = {
    "N":       ["H01"],
    "CA":      ["HA"],
    "CB":      ["HB2", "HB3"],
    "CZ2":     ["HZ22", "HZ23"],
    "OI2":     ["H1A"],
    "NK7":     ["HK7"],
    "CL7":     ["HL72", "HL73"],
    "CM7":     ["HM72", "HM73"],
    "NX8":     ["HX8"],
    "CO8":     ["HO82", "HO83"],
    "CP8":     ["HP82", "HP83"],
    "CQ2":     ["HQ2"],
    "CQ4":     ["HQ41", "HQ42", "HQ43"],
    "CQ5":     ["HQ51", "HQ52", "HQ53"],
    "S01":     ["H02"],
    "ACE_CH3": ["ACE_H1", "ACE_H2", "ACE_H3"],
    "NME_N":   ["NME_H"],
    "NME_CH3": ["NME_H1", "NME_H2", "NME_H3"],
}

HOLO_RESIDUE_ATOM_ORDER = [
    "N", "H01", "CA", "HA", "C", "O",
    "CB", "HB2", "HB3",
    "OG", "PD", "OE1", "OE2", "OE3",
    "CZ2", "HZ22", "HZ23",
    "CH2",
    "CQ4", "HQ41", "HQ42", "HQ43",
    "CQ5", "HQ51", "HQ52", "HQ53",
    "CQ2", "HQ2",
    "OI2", "H1A",
    "CI6", "OK6", "NK7", "HK7",
    "CL7", "HL72", "HL73",
    "CM7", "HM72", "HM73",
    "CN7", "OX7", "NX8", "HX8",
    "CO8", "HO82", "HO83",
    "CP8", "HP82", "HP83",
    "S01", "H02",
]

HOLO_CONFIG = {
    "name":                 "holo",
    "heavy_atoms":          HOLO_HEAVY_ATOMS,
    "bonds":                HOLO_BONDS,
    "h_names_by_parent":    HOLO_H_NAMES,
    "residue_atom_order":   HOLO_RESIDUE_ATOM_ORDER,
    "n_ppt_heavy":          27,
    "expected_total_atoms": 64,
    "formal_charge":        -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# CYS: phosphopantetheine–cysteine amide  (33 PPT heavy atoms + 5 caps = 38)
#
# Connectivity relative to holo:
#   S01 removed; CP8 now connects to NR8 (amide N)
#   …CO8–CP8–NR8(H)–CS8(=OT8)–CT9(H)(NY9·H₂)–CY10(H₂)–SF10(H)
# ─────────────────────────────────────────────────────────────────────────────
CYS_HEAVY_ATOMS = [
    # PPT residue heavy atoms (indices 0–32)
    ("N",       7,  0),   # 0
    ("CA",      6,  0),   # 1
    ("C",       6,  0),   # 2
    ("O",       8,  0),   # 3
    ("CB",      6,  0),   # 4
    ("OG",      8,  0),   # 5
    ("OE1",     8,  0),   # 6
    ("OE2",     8,  0),   # 7
    ("OE3",     8, -1),   # 8  phosphate [O⁻]
    ("CZ2",     6,  0),   # 9
    ("CH2",     6,  0),   # 10
    ("CI6",     6,  0),   # 11
    ("OI2",     8,  0),   # 12
    ("NK7",     7,  0),   # 13
    ("OK6",     8,  0),   # 14
    ("CL7",     6,  0),   # 15
    ("CM7",     6,  0),   # 16
    ("CN7",     6,  0),   # 17
    ("PD",     15,  0),   # 18
    ("NX8",     7,  0),   # 19
    ("OX7",     8,  0),   # 20
    ("CO8",     6,  0),   # 21
    ("CP8",     6,  0),   # 22
    ("CQ2",     6,  0),   # 23
    ("CQ4",     6,  0),   # 24
    ("CQ5",     6,  0),   # 25
    # Cys extension (replaces S01 at index 26)
    ("NR8",     7,  0),   # 26  amide N linking pantetheine to cysteine
    ("CS8",     6,  0),   # 27  cysteine carbonyl C (thioester → amide C=O)
    ("OT8",     8,  0),   # 28  cysteine carbonyl O
    ("CT9",     6,  0),   # 29  cysteine alpha-C
    ("NY9",     7,  0),   # 30  cysteine alpha-amino N (free NH₂)
    ("CY10",    6,  0),   # 31  cysteine beta-C
    ("SF10",   16,  0),   # 32  cysteine thiol S
    # ACE/NME caps (indices 33–37)
    ("ACE_C",   6,  0),   # 33
    ("ACE_O",   8,  0),   # 34
    ("ACE_CH3", 6,  0),   # 35
    ("NME_N",   7,  0),   # 36
    ("NME_CH3", 6,  0),   # 37
]

CYS_BONDS = [
    (0,  1,  _S),   # N–CA
    (1,  2,  _S),   # CA–C
    (2,  3,  _D),   # C=O
    (1,  4,  _S),   # CA–CB
    (4,  5,  _S),   # CB–OG
    (5,  18, _S),   # OG–PD
    (18, 6,  _D),   # PD=OE1
    (18, 7,  _S),   # PD–OE2
    (18, 8,  _S),   # PD–OE3
    (7,  9,  _S),   # OE2–CZ2
    (9,  10, _S),   # CZ2–CH2
    (10, 23, _S),   # CH2–CQ2
    (10, 24, _S),   # CH2–CQ4
    (10, 25, _S),   # CH2–CQ5
    (23, 11, _S),   # CQ2–CI6
    (23, 12, _S),   # CQ2–OI2
    (11, 14, _D),   # CI6=OK6
    (11, 13, _S),   # CI6–NK7
    (13, 15, _S),   # NK7–CL7
    (15, 16, _S),   # CL7–CM7
    (16, 17, _S),   # CM7–CN7
    (17, 20, _D),   # CN7=OX7
    (17, 19, _S),   # CN7–NX8
    (19, 21, _S),   # NX8–CO8
    (21, 22, _S),   # CO8–CP8
    # Cys extension
    (22, 26, _S),   # CP8–NR8
    (26, 27, _S),   # NR8–CS8
    (27, 28, _D),   # CS8=OT8
    (27, 29, _S),   # CS8–CT9
    (29, 30, _S),   # CT9–NY9
    (29, 31, _S),   # CT9–CY10
    (31, 32, _S),   # CY10–SF10
    # ACE cap
    (33, 34, _D),   # ACE_C=ACE_O
    (33, 35, _S),   # ACE_C–ACE_CH3
    (33, 0,  _S),   # ACE_C–N
    # NME cap
    (2,  36, _S),   # C–NME_N
    (36, 37, _S),   # NME_N–NME_CH3
]

CYS_H_NAMES = {
    "N":       ["H01"],
    "CA":      ["HA"],
    "CB":      ["HB2", "HB3"],
    "CZ2":     ["HZ22", "HZ23"],
    "OI2":     ["H1A"],
    "NK7":     ["HK7"],
    "CL7":     ["HL72", "HL73"],
    "CM7":     ["HM72", "HM73"],
    "NX8":     ["HX8"],
    "CO8":     ["HO82", "HO83"],
    "CP8":     ["HP82", "HP83"],
    "CQ2":     ["HQ2"],
    "CQ4":     ["HQ41", "HQ42", "HQ43"],
    "CQ5":     ["HQ51", "HQ52", "HQ53"],
    # Cys extension (no H on CS8, OT8)
    "NR8":     ["HR8"],
    "CT9":     ["HT9"],
    "NY9":     ["HY91", "HY92"],
    "CY10":    ["HY10", "HY11"],
    "SF10":    ["HF10"],
    # Caps
    "ACE_CH3": ["ACE_H1", "ACE_H2", "ACE_H3"],
    "NME_N":   ["NME_H"],
    "NME_CH3": ["NME_H1", "NME_H2", "NME_H3"],
}

CYS_RESIDUE_ATOM_ORDER = [
    "N", "H01", "CA", "HA", "C", "O",
    "CB", "HB2", "HB3",
    "OG", "PD", "OE1", "OE2", "OE3",
    "CZ2", "HZ22", "HZ23",
    "CH2",
    "CQ4", "HQ41", "HQ42", "HQ43",
    "CQ5", "HQ51", "HQ52", "HQ53",
    "CQ2", "HQ2",
    "OI2", "H1A",
    "CI6", "OK6", "NK7", "HK7",
    "CL7", "HL72", "HL73",
    "CM7", "HM72", "HM73",
    "CN7", "OX7", "NX8", "HX8",
    "CO8", "HO82", "HO83",
    "CP8", "HP82", "HP83",
    # Cys extension
    "NR8", "HR8",
    "CS8", "OT8",
    "CT9", "HT9", "NY9", "HY91", "HY92",
    "CY10", "HY10", "HY11",
    "SF10", "HF10",
]

CYS_CONFIG = {
    "name":                 "cys",
    "heavy_atoms":          CYS_HEAVY_ATOMS,
    "bonds":                CYS_BONDS,
    "h_names_by_parent":    CYS_H_NAMES,
    "residue_atom_order":   CYS_RESIDUE_ATOM_ORDER,
    "n_ppt_heavy":          33,
    "expected_total_atoms": 76,
    "formal_charge":        -1,
}

VARIANTS: dict[str, dict[str, Any]] = {"holo": HOLO_CONFIG, "cys": CYS_CONFIG}


# ============================================================================
# HELPERS
# ============================================================================

def _sym(name: str) -> str:
    """Element from PDB atom name: first alphabetic character (AMBER convention)."""
    alpha = "".join(c for c in name if c.isalpha())
    return alpha[0].upper() if alpha else "C"


def _atom_type_info(pdb_name: str, ff: str) -> tuple[str, str, str, float]:
    ff_types = PROTEIN_FF_TYPES[ff]
    if pdb_name in ff_types:
        tname, cname, mass = ff_types[pdb_name]
        return tname, cname, _sym(pdb_name), mass
    sym = _sym(pdb_name)
    tname = f"PPT_{pdb_name}"
    return tname, tname, sym, _ELEMENT_MASS.get(sym, 12.011)


def _fmt(value: float, decimals: int = 6) -> str:
    return f"{value:.{decimals}f}"


def _prettify(elem: ET.Element) -> str:
    raw = ET.tostring(elem, encoding="unicode")
    lines = minidom.parseString(raw).toprettyxml(indent="  ").split("\n")
    lines = [l for l in lines if not l.strip().startswith("<?xml") and l.strip()]
    return "\n".join(lines)


# ============================================================================
# PHASE 1: Build trimer and compute AM1-BCC charges
# ============================================================================

def _build_rdkit_mol(cfg: dict[str, Any]) -> Chem.Mol:
    rwmol = Chem.RWMol()
    for name, atomic_num, formal_charge in cfg["heavy_atoms"]:
        atom = Chem.Atom(atomic_num)
        atom.SetFormalCharge(formal_charge)
        rwmol.AddAtom(atom)
    for idx_a, idx_b, bond_type in cfg["bonds"]:
        rwmol.AddBond(idx_a, idx_b, bond_type)
    Chem.SanitizeMol(rwmol)
    return rwmol.GetMol()


def _build_index_name_map(mol_with_h: Chem.Mol, cfg: dict[str, Any]) -> dict[int, str]:
    idx_to_name = {i: name for i, (name, _, _) in enumerate(cfg["heavy_atoms"])}
    h_count = {}
    for atom in mol_with_h.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        h_idx = atom.GetIdx()
        parent_idx = list(atom.GetNeighbors())[0].GetIdx()
        parent_name = cfg["heavy_atoms"][parent_idx][0]
        count = h_count.get(parent_idx, 0)
        h_names = cfg["h_names_by_parent"].get(parent_name, [])
        idx_to_name[h_idx] = h_names[count] if count < len(h_names) else f"{parent_name}_H{count+1}"
        h_count[parent_idx] = count + 1
    return idx_to_name


def phase1_build_trimer(cfg: dict[str, Any], skip_charges: bool = False) -> tuple[Interchange, dict[int, str]]:
    variant = cfg["name"]
    out_dir = DATA_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)   # parents=True: data/ may not exist yet

    cache = out_dir / "trimer_interchange.json"
    if skip_charges:
        if not cache.exists():
            raise FileNotFoundError(
                f"--skip-charges requires a cached interchange; run without it first.\n"
                f"  Missing: {cache}"
            )
        print(f"\n  Loading cached interchange from {cache} ...")
        with open(cache) as f:
            interchange = Interchange.model_validate_json(f.read())
        with open(out_dir / "atom_index_to_name.json") as f:
            idx_to_name = {int(k): v for k, v in json.load(f).items()}
        return interchange, idx_to_name

    print(f"\n  Building {cfg['expected_total_atoms']}-atom trimer "
          f"({cfg['n_ppt_heavy']} PPT heavy atoms)...")

    mol_heavy = _build_rdkit_mol(cfg)
    n_heavy = mol_heavy.GetNumAtoms()
    fc = sum(a.GetFormalCharge() for a in mol_heavy.GetAtoms())
    assert fc == cfg["formal_charge"], f"Formal charge {fc}, expected {cfg['formal_charge']}"

    mol = Chem.AddHs(mol_heavy)
    n_total = mol.GetNumAtoms()
    assert n_total == cfg["expected_total_atoms"], \
        f"Expected {cfg['expected_total_atoms']} atoms, got {n_total}"
    print(f"  Heavy: {n_heavy}  Total after AddHs: {n_total}  Charge: {fc:+d}")

    idx_to_name = _build_index_name_map(mol, cfg)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError("Conformer generation failed")
    AllChem.MMFFOptimizeMolecule(mol)

    offmol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    for atom in offmol.atoms:
        atom.name = idx_to_name[atom.molecule_atom_index]

    print("  Assigning AM1-BCC charges (may take several minutes)...")
    offmol.assign_partial_charges(partial_charge_method="am1bcc")
    charge_sum = sum(float(c.magnitude) for c in offmol.partial_charges)
    print(f"  Charge sum: {charge_sum:+.6f}  (expected: {cfg['formal_charge']:+.1f})")

    ff = OFFForceField("openff-2.3.0.offxml")
    topology = Topology.from_molecules([offmol])
    interchange = ff.create_interchange(topology, charge_from_molecules=[offmol])
    print("  Interchange created")

    with open(out_dir / "atom_index_to_name.json", "w") as f:
        json.dump({str(k): v for k, v in idx_to_name.items()}, f, indent=2)
    with open(out_dir / "h_names_by_parent.json", "w") as f:
        json.dump(cfg["h_names_by_parent"], f, indent=2)
    writer = Chem.SDWriter(str(out_dir / "trimer_charged.sdf"))
    writer.write(offmol.to_rdkit())
    writer.close()
    with open(out_dir / "trimer_interchange.json", "w") as f:
        f.write(interchange.json())

    return interchange, idx_to_name


# ============================================================================
# PHASE 2: Extract FF parameters from Interchange
# ============================================================================

def _is_relevant(indices: tuple[int, ...], cap_indices: set[int], new_type_indices: set[int]) -> bool:
    s = set(indices)
    return not (s & cap_indices) and bool(s & new_type_indices)


def phase2_extract_parameters(
    cfg: dict[str, Any],
    ff: str,
    interchange: Interchange | None = None,
    idx_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    variant = cfg["name"]
    charge_dir = DATA_DIR / variant          # FF-independent cache
    out_dir    = DATA_DIR / variant / ff     # FF-specific outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    if interchange is None:
        with open(charge_dir / "trimer_interchange.json") as f:
            interchange = Interchange.model_validate_json(f.read())
    if idx_to_name is None:
        with open(charge_dir / "atom_index_to_name.json") as f:
            idx_to_name = {int(k): v for k, v in json.load(f).items()}

    ppt_atom_indices = {i for i, n in idx_to_name.items()
                        if not (n.startswith("ACE_") or n.startswith("NME_"))}
    cap_atom_indices = {i for i in idx_to_name if i not in ppt_atom_indices}

    idx_to_type  = {}
    idx_to_class = {}
    for idx, name in idx_to_name.items():
        if name.startswith("ACE_") or name.startswith("NME_"):
            continue
        t, c, _, _ = _atom_type_info(name, ff)
        idx_to_type[idx]  = t
        idx_to_class[idx] = c

    # "Borrowed from the protein FF" is a property of the atom's TYPE, so test
    # against the protein type names — not the class names, which differ for ff14sb.
    protein_type_names = PROTEIN_TYPE_NAMES[ff]
    new_type_indices = {i for i in ppt_atom_indices
                        if idx_to_type[i] not in protein_type_names}

    # add_constrained_forces=True keeps H-X bonds in HarmonicBondForce so their
    # equilibrium lengths are available for extraction and written to ppt_residue.xml.
    # Without this, OpenMM has no length to constrain PPT H atoms to and they float.
    system = interchange.to_openmm_system(add_constrained_forces=True)
    bond_force = angle_force = torsion_force = nb_force = None
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):      bond_force    = force
        elif isinstance(force, openmm.HarmonicAngleForce):   angle_force   = force
        elif isinstance(force, openmm.PeriodicTorsionForce): torsion_force = force
        elif isinstance(force, openmm.NonbondedForce):       nb_force      = force

    # Charges: read from Interchange, renormalize to exact -1.0
    ic_charges = {}
    for key, val in interchange["Electrostatics"].key_map.items():
        atom_idx = key.atom_indices[0]
        ic_charges[atom_idx] = float(
            interchange["Electrostatics"].potentials[val].parameters["charge"].magnitude
        )
    ppt_charge_sum = sum(ic_charges[i] for i in ppt_atom_indices)
    correction = (-1.0 - ppt_charge_sum) / len(ppt_atom_indices)
    if abs(ppt_charge_sum - (-1.0)) > 1e-6:
        for i in ppt_atom_indices:
            ic_charges[i] += correction
        print(f"  Charges renormalized by {correction:+.8f} e/atom; "
              f"sum → {sum(ic_charges[i] for i in ppt_atom_indices):+.6f}")
    charges = ic_charges

    # vdW
    vdw_by_class = {}
    for idx in new_type_indices:
        _, sig, eps = nb_force.getParticleParameters(idx)
        cls = idx_to_class[idx]
        vdw_by_class[cls] = {
            "sigma_nm":      sig.value_in_unit(mm_unit.nanometer),
            "epsilon_kJmol": eps.value_in_unit(mm_unit.kilojoule_per_mole),
        }

    def is_rel(idxs): return _is_relevant(idxs, cap_atom_indices, new_type_indices)
    def classes(idxs): return tuple(idx_to_class[i] for i in idxs)

    # Bonds
    bond_params = {}
    for i in range(bond_force.getNumBonds()):
        a, b, length, k = bond_force.getBondParameters(i)
        if not is_rel((a, b)): continue
        ca, cb = classes((a, b))
        key = (ca, cb) if ca <= cb else (cb, ca)
        bond_params[key] = {
            "k":      k.value_in_unit(mm_unit.kilojoule_per_mole / mm_unit.nanometer**2),
            "length": length.value_in_unit(mm_unit.nanometer),
        }

    # Angles
    angle_params = {}
    for i in range(angle_force.getNumAngles()):
        a, b, c, angle, k = angle_force.getAngleParameters(i)
        if not is_rel((a, b, c)): continue
        ca, cb, cc = classes((a, b, c))
        key = (ca, cb, cc) if ca <= cc else (cc, cb, ca)
        angle_params[key] = {
            "angle": angle.value_in_unit(mm_unit.radian),
            "k":     k.value_in_unit(mm_unit.kilojoule_per_mole / mm_unit.radian**2),
        }

    # Torsions
    torsion_params = defaultdict(list)
    for i in range(torsion_force.getNumTorsions()):
        a, b, c, d, periodicity, phase, k = torsion_force.getTorsionParameters(i)
        if not is_rel((a, b, c, d)): continue
        ca, cb, cc, cd = classes((a, b, c, d))
        fwd, rev = (ca, cb, cc, cd), (cd, cc, cb, ca)
        key = fwd if fwd <= rev else rev
        term = {
            "periodicity": int(periodicity),
            "k":           k.value_in_unit(mm_unit.kilojoule_per_mole),
            "phase":       phase.value_in_unit(mm_unit.radian),
        }
        if term not in torsion_params[key]:
            torsion_params[key].append(term)

    # Residue atom list
    name_to_idx = {v: k for k, v in idx_to_name.items() if k in ppt_atom_indices}
    residue_atoms = []
    for atom_name in cfg["residue_atom_order"]:
        idx = name_to_idx[atom_name]
        tname, cname, sym, mass = _atom_type_info(atom_name, ff)
        residue_atoms.append({
            "name": atom_name, "type": tname, "class": cname,
            "element": sym, "mass": mass,
            "charge": round(charges[idx], 6), "idx": idx,
        })

    new_type_registry = {
        a["type"]: {"class": a["class"], "element": a["element"], "mass": a["mass"]}
        for a in residue_atoms if a["type"] not in protein_type_names
    }

    # Residue bond connectivity from topology (includes H-X bonds)
    top = interchange.topology
    residue_bonds_conn = []
    for bond in top.bonds:
        a_idx = top.atom_index(bond.atom1)
        b_idx = top.atom_index(bond.atom2)
        if a_idx in ppt_atom_indices and b_idx in ppt_atom_indices:
            na, nb_ = idx_to_name[a_idx], idx_to_name[b_idx]
            residue_bonds_conn.append((na, nb_) if na <= nb_ else (nb_, na))
    residue_bonds_conn = sorted(set(residue_bonds_conn))

    print(f"  {len(new_type_registry)} new types, "
          f"{len(residue_atoms)} residue atoms, "
          f"{len(residue_bonds_conn)} internal bonds")
    print(f"  {len(bond_params)} bond params, "
          f"{len(angle_params)} angle params, "
          f"{len(torsion_params)} torsion quadruples")

    params_out = {
        "metadata": {
            "variant":       variant,
            "ff_source":     "openff-2.3.0.offxml (Sage)",
            "charge_method": "AM1-BCC",
            "coulomb14scale": 0.8333333333,
            "lj14scale":      0.5,
        },
        "new_atom_types": new_type_registry,
        "vdw":  {k: v for k, v in sorted(vdw_by_class.items())},
        "residue_atoms":             residue_atoms,
        "residue_bonds_connectivity": residue_bonds_conn,
        "bonds":  {f"{c1}~{c2}": p     for (c1, c2), p     in sorted(bond_params.items())},
        "angles": {f"{c1}~{c2}~{c3}": p for (c1, c2, c3), p in sorted(angle_params.items())},
        "proper_torsions": {
            f"{c1}~{c2}~{c3}~{c4}": terms
            for (c1, c2, c3, c4), terms in sorted(torsion_params.items())
        },
    }

    with open(out_dir / "parameters.json", "w") as f:
        json.dump(params_out, f, indent=2)

    return params_out


# ============================================================================
# PHASE 3: Write OpenMM ForceField XML
# ============================================================================

def phase3_write_xml(cfg: dict[str, Any], ff: str, params: dict[str, Any] | None = None) -> None:
    variant = cfg["name"]
    out_dir = DATA_DIR / variant / ff
    out_dir.mkdir(parents=True, exist_ok=True)

    if params is None:
        with open(out_dir / "parameters.json") as f:
            params = json.load(f)

    meta           = params["metadata"]
    new_types      = params["new_atom_types"]
    vdw            = params["vdw"]
    residue_atoms  = params["residue_atoms"]
    res_bonds_conn = params["residue_bonds_connectivity"]
    bond_params    = params["bonds"]
    angle_params   = params["angles"]
    torsion_params = params["proper_torsions"]

    protein_classes = PROTEIN_CLASS_NAMES[ff]

    FF_LABEL = {
        "ff14sb": "ff14SB (amber14-all.xml)",
        "ff19sb": "ff19SB (amber19/protein.ff19SB.xml)",
    }
    root = ET.Element("ForceField")
    root.append(ET.Comment(
        f" Custom parameters for PPT-modified serine ({variant} variant, PCP1)\n"
        f"  Source: OpenFF Sage 2.3.0, AM1-BCC charges\n"
        f"  Backbone/CB use {FF_LABEL[ff]}.\n"
        f"  coulomb14scale={meta['coulomb14scale']:.7f}  lj14scale={meta['lj14scale']}\n"
    ))

    atom_types_el = ET.SubElement(root, "AtomTypes")
    for type_name, info in sorted(new_types.items()):
        ET.SubElement(atom_types_el, "Type", attrib={
            "name": type_name, "class": info["class"],
            "element": info["element"], "mass": _fmt(info["mass"], 5),
        })

    residues_el = ET.SubElement(root, "Residues")
    res_el = ET.SubElement(residues_el, "Residue", name="PPT")
    for atom in residue_atoms:
        ET.SubElement(res_el, "Atom", attrib={
            "name": atom["name"], "type": atom["type"],
            "charge": _fmt(atom["charge"], 6),
        })
    for n1, n2 in res_bonds_conn:
        ET.SubElement(res_el, "Bond", attrib={"atomName1": n1, "atomName2": n2})
    ET.SubElement(res_el, "ExternalBond", atomName="N")
    ET.SubElement(res_el, "ExternalBond", atomName="C")

    bond_force_el = ET.SubElement(root, "HarmonicBondForce")
    for key_str, p in sorted(bond_params.items()):
        c1, c2 = key_str.split("~")
        if c1 in protein_classes and c2 in protein_classes:
            continue
        ET.SubElement(bond_force_el, "Bond", attrib={
            "class1": c1, "class2": c2,
            "length": _fmt(p["length"], 6), "k": _fmt(p["k"], 4),
        })

    angle_force_el = ET.SubElement(root, "HarmonicAngleForce")
    for key_str, p in sorted(angle_params.items()):
        c1, c2, c3 = key_str.split("~")
        if c1 in protein_classes and c2 in protein_classes and c3 in protein_classes:
            continue
        ET.SubElement(angle_force_el, "Angle", attrib={
            "class1": c1, "class2": c2, "class3": c3,
            "angle": _fmt(p["angle"], 6), "k": _fmt(p["k"], 4),
        })

    torsion_force_el = ET.SubElement(root, "PeriodicTorsionForce")
    for key_str, terms in sorted(torsion_params.items()):
        c1, c2, c3, c4 = key_str.split("~")
        if all(c in protein_classes for c in (c1, c2, c3, c4)):
            continue
        attrib = {"class1": c1, "class2": c2, "class3": c3, "class4": c4}
        for i, term in enumerate(terms, start=1):
            attrib[f"periodicity{i}"] = str(term["periodicity"])
            attrib[f"k{i}"]           = _fmt(term["k"], 6)
            attrib[f"phase{i}"]       = _fmt(term["phase"], 6)
        ET.SubElement(torsion_force_el, "Proper", attrib=attrib)

    nb_force_el = ET.SubElement(root, "NonbondedForce", attrib={
        "coulomb14scale": _fmt(meta["coulomb14scale"], 10),
        "lj14scale":      _fmt(meta["lj14scale"], 1),
    })
    # Charges must come from the <Residue> template, exactly as AMBER's own XMLs do.
    # An explicit charge= on these <Atom> elements takes precedence over the residue
    # attribute in OpenMM, so emitting charge="0" here would zero out every PPT_*
    # atom — silently neutralizing the whole phosphopantetheine arm.
    ET.SubElement(nb_force_el, "UseAttributeFromResidue", name="charge")
    for class_name, vdw_params in sorted(vdw.items()):
        ET.SubElement(nb_force_el, "Atom", attrib={
            "type":    class_name,
            "sigma":   _fmt(vdw_params["sigma_nm"], 7),
            "epsilon": _fmt(vdw_params["epsilon_kJmol"], 7),
        })

    out_path = out_dir / "ppt_residue.xml"
    out_path.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + _prettify(root) + "\n")


    # Summary
    n_bonds   = sum(1 for k in bond_params   if not all(c in protein_classes for c in k.split("~")))
    n_angles  = sum(1 for k in angle_params  if not all(c in protein_classes for c in k.split("~")))
    n_torsions = sum(1 for k in torsion_params if not all(c in protein_classes for c in k.split("~")))
    h_atoms    = {a["name"] for a in residue_atoms if a["element"] == "H"}
    total_charge = sum(a["charge"] for a in residue_atoms)

    print(f"  Wrote {out_path}  ({out_path.stat().st_size} bytes)")
    print(f"  AtomTypes: {len(new_types)}  |  "
          f"Residue: {len(residue_atoms)} atoms ({len(residue_atoms)-len(h_atoms)} heavy, {len(h_atoms)} H)")
    print(f"  Bonds: {n_bonds}  Angles: {n_angles}  Torsions: {n_torsions}")
    print(f"  Charge sum: {total_charge:+.6f} e  "
          f"({'OK' if abs(total_charge + 1) < 0.001 else 'WARNING'})")


# ============================================================================
# MAIN
# ============================================================================

def run_variant(cfg: dict[str, Any], ffs: list[str], skip_charges: bool = False) -> None:
    variant = cfg["name"]
    print("\n" + "=" * 60)
    print(f"VARIANT: {variant}")
    print("=" * 60)

    # Phase 1: build trimer and compute AM1-BCC charges — done once per variant
    # regardless of how many FFs are requested (charges are FF-independent).
    print("\n[Phase 1] Build trimer + AM1-BCC charges")
    interchange, idx_to_name = phase1_build_trimer(cfg, skip_charges=skip_charges)

    for ff in ffs:
        print(f"\n── FF: {ff} ──")
        print("\n[Phase 2] Extract parameters")
        params = phase2_extract_parameters(cfg, ff, interchange, idx_to_name)

        print("\n[Phase 3] Write XML")
        phase3_write_xml(cfg, ff, params)

        print(f"\n  Done → data/{variant}/{ff}/ppt_residue.xml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameterize PPT-modified serine variants for PCP1"
    )
    parser.add_argument(
        "--variant", choices=["holo", "cys", "all"], default="all",
        help="Which variant to parameterize (default: all)"
    )
    parser.add_argument(
        "--ff",
        type=lambda s: s.lower(),
        choices=["ff14sb", "ff19sb", "all"], default="all",
        help="Protein FF to generate XML for: ff14sb | ff19sb | all (default: all)",
    )
    parser.add_argument(
        "--skip-charges", action="store_true",
        help="Load cached trimer_interchange.json instead of rerunning AM1-BCC"
    )
    args = parser.parse_args()

    all_ffs = ["ff14sb", "ff19sb"]
    ffs = all_ffs if args.ff == "all" else [args.ff]

    to_run = list(VARIANTS.values()) if args.variant == "all" else [VARIANTS[args.variant]]
    for cfg in to_run:
        run_variant(cfg, ffs=ffs, skip_charges=args.skip_charges)

    print("\n" + "=" * 60)
    print("Parameterization complete.")
    print("=" * 60)
