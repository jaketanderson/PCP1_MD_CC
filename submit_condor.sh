#!/usr/bin/env bash
#
# submit_condor.sh - queue PCP1 MD replicates on NMRbox via HTCondor.
#
# One job per (state, ff, water, seed) replicate, throttled so only a few run
# at once. All bookkeeping lands under <script_dir>/workspace.
#
#   ./submit_condor.sh --test       ~10 ps per job, then validate each system
#   ./submit_condor.sh              the real matrix (550 ns per replicate)
#   ./submit_condor.sh --dry-run    write the submit file, do not queue
#   ./submit_condor.sh --states holo --waters opc --seeds "1 2"
#
# Options:
#   --test                 short run + validation instead of production
#   --dry-run              generate everything but skip condor_submit
#   --force                resubmit even if production.dcd already exists
#   --states/--ffs/--waters/--seeds "a b"   override the matrix
#   --max-running N        concurrent job cap (default 4)
#   --max-hours H          wall-clock budget per segment (default 6)
#   --checkpoint-ns X      checkpoint interval (default 10)
#   --runtime-ns/--equil-ns X               passed through to simulate.py
#   --cpus/--memory/--gpu-capability        resource requests
#
# This script is also its own job wrapper: HTCondor re-invokes it as
#   submit_condor.sh --exec <state> <ff> <water> <seed> <runtag> [flags...]
# so there is only one file to keep in sync.
#
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# ── Defaults ────────────────────────────────────────────────────────────────
STATES=(apo holo cys-loaded)
FFS=(ff14sb ff19sb)
WATERS=(tip3p opc)
SEEDS=(1 2 3)

REQUEST_CPUS=2
REQUEST_MEMORY="2GB"
REQUEST_DISK="1GB"
REQUEST_GPUS=1
GPU_CAPABILITY="8.9"
MAX_RUNNING=6

# NMRbox advises against jobs over 8 h, so each invocation runs at most this long
# then checkpoints and asks to be requeued (see on_exit_remove in the submit file).
MAX_HOURS="6"
CHECKPOINT_NS="10"
CHECKPOINT_EXIT_CODE=85
# Distinct from 85 so the two are distinguishable in the logs: 85 means "work
# done, checkpointed, continue me"; 86 means "this slot's GPU was unusable,
# nothing was done, please reschedule".
GPU_RETRY_EXIT_CODE=86

CONDA_ENV="${CONDA_ENV:-openmm}"

RUNTIME_NS=""          # empty => simulate.py default (550 ns)
EQUIL_NS=""            # empty => simulate.py default (1.0 ns)
# simulate.py reports every 20 ps, so a test shorter than that produces empty
# logs and an empty trajectory. 0.05 ns yields two reports from each stage.
TEST_RUNTIME_NS="0.05"
TEST_EQUIL_NS="0.05"

WORKSPACE_ROOT="$SCRIPT_DIR/workspace"
TEST_MODE=0
DRY_RUN=0
FORCE=0

sha256() { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; }

# ════════════════════════════════════════════════════════════════════════════
# JOB SIDE - invoked by HTCondor, not by you
# ════════════════════════════════════════════════════════════════════════════
if [[ "${1:-}" == "--exec" ]]; then
    shift
    state="$1"; ff="$2"; water="$3"; seed="$4"; runtag="$5"; shift 5
    extra_flags=("$@")

    cd "$SCRIPT_DIR"

    ws_root="$WORKSPACE_ROOT"
    [[ "$runtag" == "test" ]] && ws_root="$WORKSPACE_ROOT/_test"
    rundir="$ws_root/$state/$ff/$water/$seed"
    metadir="$rundir/condor"
    mkdir -p "$metadir"
    info="$metadir/job_info.txt"

    case "$state" in
        apo)        pdb="apo.pdb";    variant="" ;;
        holo)       pdb="holo.pdb";   variant="holo" ;;
        cys-loaded) pdb="loaded.pdb"; variant="cys" ;;
        *) echo "FATAL: unknown state '$state'" >&2; exit 64 ;;
    esac

    # A 550 ns replicate runs as a chain of requeued segments, so append rather
    # than overwrite: job_info.txt accumulates one block per segment. (Condor's
    # own condor.out/err are recreated each run and only show the latest.)
    segment=$(( $(grep -c '^# === segment' "$info" 2>/dev/null || echo 0) + 1 ))
    {
        echo ""
        echo "# === segment $segment  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        echo "run_tag           = $runtag"
        echo "state             = $state"
        echo "ff                = $ff"
        echo "water             = $water"
        echo "seed              = $seed"
        echo "input_pdb         = $pdb"
        echo "extra_flags       = ${extra_flags[*]:-(none)}"
        echo "condor_cluster    = ${_CONDOR_JOB_AD:+$(awk -F' = ' '/^ClusterId/{print $2}' "$_CONDOR_JOB_AD" 2>/dev/null)}"
        echo "condor_proc       = ${_CONDOR_JOB_AD:+$(awk -F' = ' '/^ProcId/{print $2}' "$_CONDOR_JOB_AD" 2>/dev/null)}"
        echo "condor_slot       = ${_CONDOR_SLOT:-unset}"
        echo "execute_host      = $(hostname -f 2>/dev/null || hostname)"
        echo "submit_dir        = $SCRIPT_DIR"
        echo "working_dir       = $(pwd)"
        echo "output_dir        = $rundir"
        echo "user              = $(id -un)"
        echo "start_time_utc    = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "conda_env         = $CONDA_ENV"
        echo "git_commit        = $(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
        echo "git_dirty_files   = $(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

        # Which GPU actually ran this, so a capability mismatch or an unexpected
        # card is visible after the fact rather than inferred from timings.
        echo "cuda_visible_dev  = ${CUDA_VISIBLE_DEVICES:-unset}"
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "gpu               = $(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | paste -sd'; ' -)"
            echo "nvidia_driver     = $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        else
            echo "gpu               = (nvidia-smi unavailable)"
        fi

        # Checksums at execution time, to diff against the submit-time manifest.
        echo ""
        echo "# input checksums (sha256, at execution time)"
        for f in simulate.py "$pdb" ${variant:+"data/$variant/$ff/ppt_residue.xml"}; do
            if [[ -f "$f" ]]; then sha256 "$f"; else echo "MISSING  $f"; fi
        done
    } >> "$info"

    # getenv=True usually carries the environment over, but be explicit so the
    # job does not depend on how it happened to be submitted.
    #
    # Do NOT use `$(conda info --base)` unguarded: a broken conda plugin prints
    # its load failure to stdout, which ends up substituted in place of the path
    # (seen on NMRbox with anaconda-anon-usage). Derive the base from $CONDA_EXE
    # where possible, and otherwise accept only output that is a real directory.
    conda_base=""
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]]; then
        conda_base="$(cd "$(dirname "$CONDA_EXE")/.." 2>/dev/null && pwd)"
    fi
    if [[ ! -f "${conda_base:-}/etc/profile.d/conda.sh" ]] \
       && command -v conda >/dev/null 2>&1; then
        # keep only the last line that looks like an absolute path
        conda_base="$(conda info --base 2>/dev/null \
                      | tr -d '\r' | awk '/^\//{p=$0} END{print p}')"
    fi

    conda_sh=""
    for base in "$conda_base" "$HOME/miniconda3" "$HOME/anaconda3" \
                "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda; do
        if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
            conda_sh="$base/etc/profile.d/conda.sh"
            conda_base="$base"
            break
        fi
    done
    [[ -n "$conda_sh" ]] \
        || { echo "FATAL: no conda.sh found (CONDA_EXE=${CONDA_EXE:-unset})" >&2; exit 78; }

    # shellcheck disable=SC1090
    . "$conda_sh"
    conda activate "$CONDA_ENV" \
        || { echo "FATAL: cannot activate conda env '$CONDA_ENV'" >&2; exit 78; }

    # `conda activate` can report success while leaving PATH untouched if the
    # shell hook is half-broken, so resolve the interpreter explicitly and fall
    # back to the env's python directly rather than silently using the wrong one.
    py="$(command -v python 2>/dev/null || true)"
    if [[ -z "$py" || -z "${CONDA_PREFIX:-}" || "$py" != "${CONDA_PREFIX:-/nonexistent}"/* ]] \
       || ! "$py" -c 'import openmm' >/dev/null 2>&1; then
        py="$conda_base/envs/$CONDA_ENV/bin/python"
        echo "NOTE: activation did not yield a usable interpreter; using $py" >&2
    fi
    if [[ ! -x "$py" ]] || ! "$py" -c 'import openmm' >/dev/null 2>&1; then
        echo "FATAL: no working python with openmm for env '$CONDA_ENV'" >&2
        echo "       conda_base=$conda_base  candidate=$py" >&2
        exit 78
    fi

    # Record OpenMM's own installation self-test before anything else. It lists the
    # platforms actually present on this node and cross-checks their energies
    # against each other, which characterises a node where CUDA is missing, broken
    # or disagreeing with the CPU platform -- and it is captured per segment, so a
    # node that starts misbehaving mid-campaign is visible after the fact.
    {
        echo ""
        echo "# openmm.testInstallation"
        "$py" -m openmm.testInstallation 2>&1 | sed 's/^/  /' || true
    } >> "$info"

    # Preflight: can OpenMM actually open a CUDA context on the GPU we were given?
    #
    # A slot can hand over a GPU that nvidia-smi queries fine but that cannot be
    # claimed -- exclusive compute mode with another context already resident, or
    # a wedged device. Without this check simulate.py dies with exit 1, which does
    # not match on_exit_remove, so HTCondor drops the job and the replicate is
    # lost. Detect it in seconds instead and ask to be rescheduled.
    preflight="$("$py" - <<'PYEOF' 2>&1
import openmm
plat = openmm.Platform.getPlatformByName("CUDA")
s = openmm.System(); s.addParticle(1.0)
ctx = openmm.Context(s, openmm.VerletIntegrator(0.001), plat, {"Precision": "mixed"})
print("OK", ctx.getPlatform().getName(),
      ctx.getPlatform().getPropertyValue(ctx, "DeviceName"))
PYEOF
    )" && preflight_rc=0 || preflight_rc=$?

    if [[ ${preflight_rc:-1} -ne 0 ]]; then
        {
            echo "cuda_preflight    = FAILED"
            echo "preflight_error   = ${preflight//$'\n'/ | }"
            echo "# diagnostics"
            if command -v nvidia-smi >/dev/null 2>&1; then
                # compute_mode is the usual culprit: an Exclusive_Process GPU that
                # already has a context cannot be claimed, even though it queries fine.
                echo "nvidia_smi_L      = $(nvidia-smi -L 2>/dev/null | paste -sd'; ' -)"
                echo "compute_mode      = $(nvidia-smi --query-gpu=compute_mode,persistence_mode --format=csv,noheader 2>/dev/null | paste -sd'; ' -)"
                echo "gpu_processes     = $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | paste -sd'; ' -)"
                echo "ecc_errors        = $(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader 2>/dev/null | paste -sd'; ' -)"
            else
                echo "nvidia_smi        = (not on PATH)"
            fi
            echo "openmm_platforms  = $("$py" -c 'import openmm;print(", ".join(openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())))' 2>&1)"
            echo "segment_result    = GPU UNAVAILABLE, requeued"
        } >> "$info"
        echo "=== CUDA preflight failed on $(hostname -f 2>/dev/null || hostname):" >&2
        echo "$preflight" >&2
        echo "=== exiting $GPU_RETRY_EXIT_CODE to be rescheduled" >&2
        exit "$GPU_RETRY_EXIT_CODE"
    fi
    echo "cuda_preflight    = ${preflight}" >> "$info"

    cmd=("$py" simulate.py --state "$state" --ff "$ff" --water "$water"
         --seed "$seed" --workspace-root "$ws_root"
         ${extra_flags[@]+"${extra_flags[@]}"})
    {
        echo ""
        echo "command           = ${cmd[*]}"
        echo "python            = $py"
        echo "conda_base        = $conda_base"
        echo "openmm_version    = $("$py" -c 'import openmm; print(openmm.__version__)' 2>/dev/null || echo '?')"
    } >> "$info"

    echo "=== ${cmd[*]}"
    set +e
    "${cmd[@]}"
    rc=$?
    set -e
    {
        echo "end_time_utc      = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "exit_code         = $rc"
    } >> "$info"
    if [[ $rc -eq $CHECKPOINT_EXIT_CODE ]]; then
        echo "segment_result    = checkpointed, requeued" >> "$info"
        echo "=== wall-clock budget reached; checkpointed and exiting $rc to requeue"
        exit $rc
    elif [[ $rc -ne 0 ]]; then
        echo "segment_result    = FAILED" >> "$info"
        echo "FAILED: simulate.py exited $rc" >&2
        exit $rc
    fi
    echo "segment_result    = completed" >> "$info"

    # Test mode validates the produced system rather than trusting exit 0.
    if [[ "$runtag" == "test" ]]; then
        echo "=== validating $rundir"
        set +e
        "$py" - "$rundir" "$state" <<'PYEOF' | tee -a "$info"
"""Validate a finished PCP1 run directory. Exits non-zero on any problem."""
import sys
from pathlib import Path

import openmm
from openmm import app, unit

rundir, state = Path(sys.argv[1]), sys.argv[2]
problems, notes = [], []

for name in ("solvated.pdb", "system.xml", "minimized.pdb", "production.log"):
    p = rundir / name
    if not p.is_file():
        problems.append(f"missing {name}")
    elif p.stat().st_size == 0:
        problems.append(f"empty {name} (run shorter than the 20 ps "
                        "reporter interval?)")

if not problems:
    pdb = app.PDBFile(str(rundir / "solvated.pdb"))
    with open(rundir / "system.xml") as fh:
        system = openmm.XmlSerializer.deserialize(fh.read())

    if system.getNumParticles() != pdb.topology.getNumAtoms():
        problems.append(f"system has {system.getNumParticles()} particles, "
                        f"topology has {pdb.topology.getNumAtoms()}")

    nbf = next(f for f in system.getForces()
               if isinstance(f, openmm.NonbondedForce))
    q = [nbf.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
         for i in range(system.getNumParticles())]
    net = sum(q)
    notes.append(f"net_charge        = {net:+.4f}")
    if abs(net) > 1e-3:
        problems.append(f"net charge {net:+.4f} is not neutral")

    ppt = [r for r in pdb.topology.residues() if r.name == "PPT"]
    if state == "apo":
        if ppt:
            problems.append("apo system unexpectedly contains a PPT residue")
    elif not ppt:
        problems.append(f"{state} system has no PPT residue")
    else:
        res = ppt[0]
        ppt_q = sum(q[a.index] for a in res.atoms())
        notes.append(f"ppt_charge        = {ppt_q:+.4f}")
        if abs(ppt_q + 1.0) > 1e-3:
            problems.append(f"PPT charge {ppt_q:+.4f} != -1 "
                            "(stray charge= in NonbondedForce?)")

        names = {a.name: a.index for a in res.atoms()}
        if "CB" not in names or "OG" not in names:
            problems.append("PPT residue lacks CB and/or OG")
        else:
            pair = {names["CB"], names["OG"]}
            hb = next(f for f in system.getForces()
                      if isinstance(f, openmm.HarmonicBondForce))
            found = None
            for i in range(hb.getNumBonds()):
                a, b, length, k = hb.getBondParameters(i)
                if {a, b} == pair:
                    found = length.value_in_unit(unit.nanometer)
                    break
            if found is None:
                problems.append("CB-OG bond absent from HarmonicBondForce "
                                "(type/class mismatch?)")
            else:
                notes.append(f"cb_og_bond_nm     = {found:.4f}")

    rows = (rundir / "production.log").read_text().strip().splitlines()
    if len(rows) < 2:
        problems.append("production.log has no data rows")
    else:
        cols = rows[-1].split(",")
        try:
            pe, temp = float(cols[2]), float(cols[5])
            notes.append(f"final_pe_kJmol    = {pe:.4e}")
            notes.append(f"final_temp_K      = {temp:.2f}")
            if pe != pe or abs(pe) > 1e12:
                problems.append(f"potential energy not finite: {pe}")
            if not 150.0 < temp < 450.0:
                problems.append(f"temperature {temp:.1f} K out of range")
        except (ValueError, IndexError):
            problems.append(f"cannot parse production.log row: {cols}")

print("# validation")
for n in notes:
    print(n)
if problems:
    print(f"validation        = FAILED ({len(problems)} problems)")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("validation        = PASSED")
PYEOF
        rc=${PIPESTATUS[0]}
        set -e
        echo "validation_exit   = $rc" >> "$info"
        [[ $rc -ne 0 ]] && { echo "VALIDATION FAILED" >&2; exit $rc; }
        echo "=== validation passed"
    fi
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
# SUBMIT SIDE
# ════════════════════════════════════════════════════════════════════════════
usage() { sed -n '2,28p' "$SCRIPT_PATH" | sed 's/^#\ \?//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)           TEST_MODE=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --force)          FORCE=1; shift ;;
        --states)         read -ra STATES <<< "$2"; shift 2 ;;
        --ffs)            read -ra FFS <<< "$2"; shift 2 ;;
        --waters)         read -ra WATERS <<< "$2"; shift 2 ;;
        --seeds)          read -ra SEEDS <<< "$2"; shift 2 ;;
        --max-running)    MAX_RUNNING="$2"; shift 2 ;;
        --max-hours)      MAX_HOURS="$2"; shift 2 ;;
        --checkpoint-ns)  CHECKPOINT_NS="$2"; shift 2 ;;
        --runtime-ns)     RUNTIME_NS="$2"; shift 2 ;;
        --equil-ns)       EQUIL_NS="$2"; shift 2 ;;
        --cpus)           REQUEST_CPUS="$2"; shift 2 ;;
        --memory)         REQUEST_MEMORY="$2"; shift 2 ;;
        --gpu-capability) GPU_CAPABILITY="$2"; shift 2 ;;
        -h|--help)        usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

cd "$SCRIPT_DIR"

if [[ $TEST_MODE -eq 1 ]]; then
    RUNTIME_NS="$TEST_RUNTIME_NS"
    EQUIL_NS="$TEST_EQUIL_NS"
    RUNTAG="test"
    WS_ROOT="$WORKSPACE_ROOT/_test"
else
    RUNTAG="prod"
    WS_ROOT="$WORKSPACE_ROOT"
fi

extra_flags=(--checkpoint-ns "$CHECKPOINT_NS" --max-hours "$MAX_HOURS")
[[ -n "$RUNTIME_NS" ]] && extra_flags+=(--runtime-ns "$RUNTIME_NS")
[[ -n "$EQUIL_NS" ]]   && extra_flags+=(--equil-ns "$EQUIL_NS")
FLAGS_STR="${extra_flags[*]}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBDIR="$WORKSPACE_ROOT/_submissions/${STAMP}_${RUNTAG}"
mkdir -p "$SUBDIR"
JOBLIST="$SUBDIR/jobs.list"
SUBFILE="$SUBDIR/pcp1.sub"

# ── Build the job list ──────────────────────────────────────────────────────
: > "$JOBLIST"
n_jobs=0; n_skipped=0
for state in "${STATES[@]}"; do
  for ff in "${FFS[@]}"; do
    for water in "${WATERS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        rundir="$WS_ROOT/$state/$ff/$water/$seed"
        if [[ $FORCE -eq 0 && -s "$rundir/production.dcd" ]]; then
            echo "  skip (production.dcd exists): $state/$ff/$water/$seed"
            n_skipped=$((n_skipped + 1)); continue
        fi
        mkdir -p "$rundir/condor"
        echo "$state,$ff,$water,$seed,$rundir/condor" >> "$JOBLIST"
        n_jobs=$((n_jobs + 1))
      done
    done
  done
done

if [[ $n_jobs -eq 0 ]]; then
    echo "Nothing to submit ($n_skipped already complete; --force to resubmit)."
    exit 0
fi

# ── Submit-time provenance ──────────────────────────────────────────────────
{
    echo "submission        = ${STAMP}_${RUNTAG}"
    echo "run_tag           = $RUNTAG"
    echo "submitted_by      = $(id -un)@$(hostname -f 2>/dev/null || hostname)"
    echo "submit_time_utc   = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "submit_dir        = $SCRIPT_DIR"
    echo "workspace_root    = $WS_ROOT"
    echo "git_commit        = $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
    echo "git_branch        = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    echo "git_dirty_files   = $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    echo "conda_env         = $CONDA_ENV"
    echo ""
    echo "states            = ${STATES[*]}"
    echo "ffs               = ${FFS[*]}"
    echo "waters            = ${WATERS[*]}"
    echo "seeds             = ${SEEDS[*]}"
    echo "jobs_queued       = $n_jobs"
    echo "jobs_skipped      = $n_skipped"
    echo "simulate_flags    = ${FLAGS_STR:-(simulate.py defaults)}"
    echo ""
    echo "request_cpus      = $REQUEST_CPUS"
    echo "request_memory    = $REQUEST_MEMORY"
    echo "request_disk      = $REQUEST_DISK"
    echo "request_gpus      = $REQUEST_GPUS"
    echo "gpu_capability    = >= $GPU_CAPABILITY"
    echo "max_running       = $MAX_RUNNING"
    echo "max_hours_segment = $MAX_HOURS"
    echo "checkpoint_ns     = $CHECKPOINT_NS"
    echo ""
    echo "# git status at submit time"
    git status --porcelain 2>/dev/null | sed 's/^/  /' || true
} > "$SUBDIR/manifest.txt"

{
    for f in simulate.py parameterize.py submit_condor.sh \
             apo.pdb holo.pdb loaded.pdb; do
        [[ -f "$f" ]] && sha256 "$f"
    done
    find data -name 'ppt_residue.xml' -o -name 'parameters.json' 2>/dev/null \
        | sort | while read -r f; do sha256 "$f"; done
} > "$SUBDIR/checksums.sha256"

# ── Submit file ─────────────────────────────────────────────────────────────
# NMRbox has a shared filesystem, so no file transfer and jobs run in place.
# +Production is deliberately unset: these jobs need only the conda env from the
# shared home directory, so they may also land on compute-only nodes.
cat > "$SUBFILE" <<EOF
# PCP1 MD - generated by submit_condor.sh at $STAMP ($RUNTAG).
# Regenerate by re-running the script; do not hand-edit.

universe                = vanilla
executable              = $SCRIPT_PATH
getenv                  = True
should_transfer_files   = NO
transfer_executable     = FALSE
initialdir              = $SCRIPT_DIR

request_cpus            = $REQUEST_CPUS
request_memory          = $REQUEST_MEMORY
request_disk            = $REQUEST_DISK
request_gpus            = $REQUEST_GPUS
requirements            = (GPUs_Capability >= $GPU_CAPABILITY)

# Cap concurrency so we do not monopolise the pool: max_materialize limits how
# many jobs exist in the queue at once, so the next appears as one finishes.
# Deliberately no on_exit_hold - held jobs keep occupying materialization slots
# and would stall the remaining queue. Failures are recorded per job instead.
max_materialize         = $MAX_RUNNING

# Segmented running: simulate.py exits $CHECKPOINT_EXIT_CODE when its wall-clock
# budget expires with work left, having just checkpointed, and the wrapper exits
# $GPU_RETRY_EXIT_CODE when the assigned GPU cannot be used at all. Keeping such
# jobs in the queue makes HTCondor run them again, so a 550 ns replicate proceeds
# as a chain of sub-8h segments and a bad GPU costs seconds rather than the whole
# replicate. Any other exit code (0 = finished, anything else = real failure)
# removes the job, so genuine failures cannot loop forever.
on_exit_remove          = (ExitCode =!= $CHECKPOINT_EXIT_CODE) && (ExitCode =!= $GPU_RETRY_EXIT_CODE)

# Record the machines this job has run on, so a retry can be steered away from
# one that just failed. Left out of requirements deliberately: with few L40S
# nodes, excluding the previous machine on every checkpoint requeue could starve
# the job. To avoid a specific bad node, append to requirements above, e.g.
#   && (TARGET.Machine =!= "lanthanum.nmrbox.org")
job_machine_attrs       = Machine
job_machine_attrs_history_length = 4

batch_name              = PCP1-$RUNTAG-$STAMP

arguments               = --exec \$(state) \$(ff) \$(water) \$(seed) $RUNTAG $FLAGS_STR
log                     = \$(metadir)/condor.log
output                  = \$(metadir)/condor.out
error                   = \$(metadir)/condor.err

queue state,ff,water,seed,metadir from $JOBLIST
EOF

# ── Report and submit ───────────────────────────────────────────────────────
cat <<EOF

Submission : ${STAMP}_${RUNTAG}
Jobs       : $n_jobs queued, $n_skipped skipped
Concurrency: max $MAX_RUNNING at a time
Resources  : $REQUEST_CPUS cpu, $REQUEST_MEMORY mem, $REQUEST_GPUS gpu (capability >= $GPU_CAPABILITY)
Sim flags  : ${FLAGS_STR:-(simulate.py defaults)}
Bookkeeping: $SUBDIR
Run output : $WS_ROOT/<state>/<ff>/<water>/<seed>/

EOF

if [[ $DRY_RUN -eq 1 ]]; then
    echo "--dry-run: not submitting. Submit file is at:"
    echo "  $SUBFILE"
    exit 0
fi

command -v condor_submit >/dev/null 2>&1 \
    || { echo "condor_submit not found - are you on a submit host?" >&2; exit 1; }

condor_submit "$SUBFILE" | tee "$SUBDIR/condor_submit.out"

cat <<EOF

Track:    condor_q -batch-name PCP1-$RUNTAG-$STAMP
Logs:     $WS_ROOT/<state>/<ff>/<water>/<seed>/condor/
Failures: grep -L 'exit_code         = 0' $WS_ROOT/*/*/*/*/condor/job_info.txt
EOF

if [[ $TEST_MODE -eq 1 ]]; then
    cat <<EOF
Validate: grep -h '^validation  ' $WS_ROOT/*/*/*/*/condor/job_info.txt | sort | uniq -c
EOF
fi
