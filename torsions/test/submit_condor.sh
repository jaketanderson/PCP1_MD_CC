#!/usr/bin/env bash
#
# submit_condor.sh - queue the PCP1 TorsionDrive scan on NMRbox via HTCondor.
#
# One job, one node: run.sh starts the TorsionDrive master plus its work_queue
# workers and drives the whole grid. The scan takes far longer than a single job
# should hold a slot, so each invocation runs for at most --max-hours, then stops
# and asks HTCondor to requeue it. TorsionDrive caches finished grid points on
# disk, so the next segment picks up where the previous one stopped and the same
# command line serves for the first run and every rerun.
#
#   ./submit_condor.sh              queue the scan
#   ./submit_condor.sh --dry-run    write the submit file, do not queue
#
# Options:
#   --dry-run              generate everything but skip condor_submit
#   --max-hours H          wall-clock budget per segment (default 8)
#   --cpus/--memory/--disk resource requests
#   --env PATH_OR_NAME     conda env to activate (default: shared torsion env)
#   --max-stall N          give up after N segments that make no progress
#   --no-port-reclaim      do not kill our own stale processes on WQ_PORT
#
# This script is also its own job wrapper: HTCondor re-invokes it as
#   submit_condor.sh --exec [flags...]
# so there is only one file to keep in sync.
#
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# ── Defaults ────────────────────────────────────────────────────────────────
# These must stay consistent with TOTAL_CORES / the worker --memory and --disk
# flags in run.sh: Condor hands out the slot, run.sh decides how to fill it, and
# nothing reconciles the two automatically.
REQUEST_CPUS=50
REQUEST_MEMORY="244GB"
REQUEST_DISK="100GB"
# No request_gpus: psi4 is CPU-only here, and asking for a GPU would restrict
# this to the handful of GPU nodes for no benefit.

# NMRbox advises against jobs over 8 h, so each segment stops at this budget and
# requeues rather than running the scan to completion in one slot.
MAX_HOURS="8"
# Subtracted from the budget so TorsionDrive gets SIGTERM, run.sh's cleanup trap
# tears down the master and workers, and we exit deliberately -- all before the
# wall-clock limit rather than being killed partway through the teardown.
# Overridable via the environment (carried to the job by getenv=True) so a short
# smoke test can use e.g. SHUTDOWN_MARGIN_S=30 ./submit_condor.sh --max-hours 0.1
SHUTDOWN_MARGIN_S="${SHUTDOWN_MARGIN_S:-300}"
REQUEUE_EXIT_CODE=85

# A requeueing job that can never progress would otherwise cycle forever, so
# stop after this many consecutive segments that leave the scratch dir unchanged.
MAX_STALL=3

# Reclaim WQ_PORT from our own stale processes before starting (see --exec).
PORT_RECLAIM=1

CONDA_ENV="${CONDA_ENV:-/public/groups/frueh/jaanderson/miniconda3/envs/torsion}"

RUN_SCRIPT="run.sh"
METADIR="$SCRIPT_DIR/condor"
DRY_RUN=0

# ════════════════════════════════════════════════════════════════════════════
# JOB SIDE - invoked by HTCondor, not by you
# ════════════════════════════════════════════════════════════════════════════
if [[ "${1:-}" == "--exec" ]]; then
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --max-hours) MAX_HOURS="$2"; shift 2 ;;
            --env)       CONDA_ENV="$2"; shift 2 ;;
            --max-stall) MAX_STALL="$2"; shift 2 ;;
            --no-port-reclaim) PORT_RECLAIM=0; shift ;;
            *) echo "FATAL: unknown --exec flag '$1'" >&2; exit 64 ;;
        esac
    done

    cd "$SCRIPT_DIR"
    mkdir -p "$METADIR"
    info="$METADIR/job_info.txt"

    # Turn arbitrary command output into an integer. Reading these counters
    # needs care: `grep -c` prints "0" *and* exits 1 when the file exists with
    # no matches, so the usual `$(... || echo 0)` idiom yields "0\n0" and a fatal
    # arithmetic error under `set -e`.
    as_int() {
        local n
        n="$("$@" 2>/dev/null | head -1 || true)"
        n="${n//[^0-9]/}"
        echo "${n:-0}"
    }

    # The scan runs as a chain of requeued segments and Condor recreates
    # condor.out/err on every one, so append here: job_info.txt is the only
    # record that spans the whole chain.
    segment=$(( $(as_int grep -c '^# === segment' "$info") + 1 ))

    # TorsionDrive's scratch folder. Counting files in it is deliberately crude:
    # any real work lands there, and it avoids depending on the layout of a state
    # file that differs between TorsionDrive versions.
    progress_fingerprint() {
        [[ -d opt_tmp ]] || { echo 0; return; }
        find opt_tmp -type f 2>/dev/null | wc -l | tr -d ' '
    }
    progress_before="$(progress_fingerprint)"

    {
        echo ""
        echo "# === segment $segment  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
        echo "execute_host      = $(hostname -f 2>/dev/null || hostname)"
        echo "condor_cluster    = ${_CONDOR_JOB_AD:+$(awk -F' = ' '/^ClusterId/{print $2}' "$_CONDOR_JOB_AD" 2>/dev/null)}"
        echo "condor_slot       = ${_CONDOR_SLOT:-unset}"
        echo "working_dir       = $(pwd)"
        echo "user              = $(id -un)"
        echo "start_time_utc    = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "conda_env         = $CONDA_ENV"
        echo "max_hours         = $MAX_HOURS"
        echo "progress_before   = $progress_before files under opt_tmp/"
        echo "git_commit        = $(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
        echo "git_dirty_files   = $(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    } >> "$info"

    # ── conda ───────────────────────────────────────────────────────────────
    # getenv=True usually carries the environment over, but be explicit so the
    # job does not depend on how it happened to be submitted.
    #
    # CONDA_ENV is normally an absolute env prefix, so derive the installation
    # root from it directly rather than trusting `conda info --base`, whose
    # output a broken plugin can pollute (seen on NMRbox with anaconda-anon-usage).
    conda_base=""
    env_prefix=""
    if [[ "$CONDA_ENV" == /* ]]; then
        env_prefix="$CONDA_ENV"
        [[ "$CONDA_ENV" == */envs/* ]] && conda_base="${CONDA_ENV%/envs/*}"
    fi
    if [[ ! -f "${conda_base:-}/etc/profile.d/conda.sh" && -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]]; then
        conda_base="$(cd "$(dirname "$CONDA_EXE")/.." 2>/dev/null && pwd)"
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
        || { echo "FATAL: no conda.sh found for env '$CONDA_ENV'" >&2; exit 78; }

    # shellcheck disable=SC1090
    . "$conda_sh"
    conda activate "$CONDA_ENV" \
        || { echo "FATAL: cannot activate conda env '$CONDA_ENV'" >&2; exit 78; }

    # `conda activate` can report success while leaving PATH untouched if the
    # shell hook is half-broken, so put the env's bin first when the tools did
    # not actually come from it.
    [[ -z "$env_prefix" ]] && env_prefix="${CONDA_PREFIX:-}"
    if [[ -n "$env_prefix" && "$(command -v psi4 2>/dev/null)" != "$env_prefix"/* ]]; then
        echo "NOTE: activation did not put $env_prefix/bin first; prepending it" >&2
        export PATH="$env_prefix/bin:$PATH"
    fi

    # ── preflight ───────────────────────────────────────────────────────────
    # Fail loudly and immediately rather than starting a master that no worker
    # can ever join: that would hang, hit the wall clock, requeue, and burn the
    # whole allocation without producing anything.
    for tool in torsiondrive-launch psi4 work_queue_worker timeout script; do
        command -v "$tool" >/dev/null 2>&1 \
            || { echo "FATAL: '$tool' not on PATH after activating $CONDA_ENV" >&2; exit 78; }
    done
    for f in "$RUN_SCRIPT" dihedrals.txt; do
        [[ -f "$f" ]] || { echo "FATAL: missing required file '$f' in $SCRIPT_DIR" >&2; exit 66; }
    done
    [[ -x "$RUN_SCRIPT" ]] || chmod +x "$RUN_SCRIPT"

    # run.sh passes -P "$(pwd)/wq.password" to every worker; without the file the
    # workers exit at startup and the master waits forever. It is only a shared
    # secret, so generate one on first use.
    if [[ ! -s wq.password ]]; then
        echo "NOTE: creating wq.password (was missing)" >&2
        umask 077
        head -c 32 /dev/urandom | base64 > wq.password
    fi

    {
        echo "torsiondrive      = $(command -v torsiondrive-launch)"
        echo "psi4              = $(command -v psi4)"
        echo "work_queue_worker = $(command -v work_queue_worker)"
    } >> "$info"

    # ── reclaim the work_queue port ─────────────────────────────────────────
    # run.sh's cleanup trap normally frees WQ_PORT, so this is for the cases it
    # cannot cover: a segment that had to be SIGKILLed after the grace period, or
    # a psi4 wedged in uninterruptible I/O that never acted on the TERM. A stale
    # listener would otherwise make the master fail to bind, and we would burn a
    # whole 8 h slot discovering that.
    #
    # Read the port out of run.sh rather than duplicating the constant.
    wq_port="$(awk -F= '/^WQ_PORT=/{print $2; exit}' "$RUN_SCRIPT" | awk '{print $1}')"
    wq_port="${wq_port//[^0-9]/}"
    : "${wq_port:=50124}"

    # DANGER: lsof ORs its selection flags unless -a is given. `lsof -u me -i :p`
    # means "owned by me OR on port p", which is every process this user owns --
    # including this job's own shell. -a is what makes it an AND. Do not remove it.
    me="$(id -un)"
    # `|| true`: lsof exits 1 when nothing matches, which is the common case
    # here (port free). Under set -euo pipefail that propagates through the
    # pipe and, because this function's result is consumed via a plain
    # assignment (`stale=$(port_pids | ...)`) rather than inside `[[ ]]`, a
    # bare 1 here would abort the whole job on the ordinary, nothing-to-do path.
    port_pids() { lsof -t -a -u "$me" -i ":$wq_port" 2>/dev/null | sort -u || true; }

    if [[ $PORT_RECLAIM -eq 1 ]] && command -v lsof >/dev/null 2>&1; then
        stale="$(port_pids | tr '\n' ' ')"
        if [[ -n "${stale// /}" ]]; then
            echo "WARNING: port $wq_port is already in use by $me; reclaiming it." >&2
            {
                echo "port_reclaim      = killing stale holders of $wq_port"
                for pid in $stale; do
                    echo "  pid $pid  $(ps -o command= -p "$pid" 2>/dev/null | cut -c1-100)"
                done
            } | tee -a "$info" >&2
            # shellcheck disable=SC2086
            kill -TERM $stale 2>/dev/null || true
            for _ in $(seq 1 15); do
                [[ -z "$(port_pids)" ]] && break
                sleep 1
            done
            survivors="$(port_pids | tr '\n' ' ')"
            if [[ -n "${survivors// /}" ]]; then
                # shellcheck disable=SC2086
                kill -KILL $survivors 2>/dev/null || true
                sleep 2
            fi
        fi

        # Anything left is not ours to kill (another user, or a process that
        # survived SIGKILL in D state). Fail fast: starting now just means
        # hanging until the budget expires and requeueing onto the same wall.
        if [[ -n "$(lsof -t -i ":$wq_port" 2>/dev/null | sort -u)" ]]; then
            echo "port_reclaim      = FAILED, $wq_port still held" >> "$info"
            echo "FATAL: port $wq_port is still in use by a process we cannot kill:" >&2
            lsof -i ":$wq_port" 2>/dev/null | sed 's/^/  /' >&2
            exit 69
        fi
        echo "port_reclaim      = $wq_port free" >> "$info"
    fi

    # ── run one segment ─────────────────────────────────────────────────────
    # `timeout` signals run.sh, whose EXIT/TERM trap tears down the master and
    # the workers; --kill-after is the backstop for a teardown that itself wedges.
    budget_s=$(awk -v h="$MAX_HOURS" 'BEGIN{printf "%d", h*3600}')
    run_s=$(( budget_s - SHUTDOWN_MARGIN_S ))
    if (( run_s <= 0 )); then
        echo "FATAL: --max-hours $MAX_HOURS leaves no time after the ${SHUTDOWN_MARGIN_S}s shutdown margin" >&2
        exit 64
    fi
    echo "segment_budget_s  = $run_s" >> "$info"

    echo "=== segment $segment: ./$RUN_SCRIPT under a ${run_s}s budget"
    set +e
    timeout --signal=TERM --kill-after="$SHUTDOWN_MARGIN_S" "$run_s" "./$RUN_SCRIPT"
    rc=$?
    set -e

    # run.sh truncates td_logs/torsiondrive.log on every start, so keep this
    # segment's logs before the next one overwrites them.
    if [[ -d td_logs ]]; then
        seg_logs="$METADIR/segments/$(printf '%03d' "$segment")"
        mkdir -p "$seg_logs"
        cp -a td_logs/. "$seg_logs"/ 2>/dev/null || true
    fi

    progress_after="$(progress_fingerprint)"
    {
        echo "end_time_utc      = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "exit_code         = $rc"
        echo "progress_after    = $progress_after files under opt_tmp/"
    } >> "$info"

    # 124 = timeout sent TERM, 137 = it had to follow up with KILL. Both mean the
    # budget expired with work left, which is the normal end of a segment.
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        stall_file="$METADIR/stall_count"
        if [[ "$progress_after" -gt "$progress_before" ]]; then
            echo 0 > "$stall_file"
        else
            stalls=$(( $(as_int cat "$stall_file") + 1 ))
            echo "$stalls" > "$stall_file"
            if (( stalls >= MAX_STALL )); then
                echo "segment_result    = STALLED ($stalls segments, no progress)" >> "$info"
                echo "FAILED: $stalls consecutive segments made no progress; not requeueing." >&2
                echo "        Check $METADIR/segments/ and td_logs/worker_*.log." >&2
                exit 75
            fi
            echo "stall_count       = $stalls" >> "$info"
        fi
        echo "segment_result    = budget reached, requeued" >> "$info"
        echo "=== wall-clock budget reached; exiting $REQUEUE_EXIT_CODE to requeue"
        exit "$REQUEUE_EXIT_CODE"
    elif [[ $rc -ne 0 ]]; then
        echo "segment_result    = FAILED" >> "$info"
        echo "FAILED: $RUN_SCRIPT exited $rc" >&2
        exit $rc
    fi

    echo "segment_result    = scan completed" >> "$info"
    echo "=== TorsionDrive finished"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
# SUBMIT SIDE
# ════════════════════════════════════════════════════════════════════════════
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --cpus)      REQUEST_CPUS="$2"; shift 2 ;;
        --memory)    REQUEST_MEMORY="$2"; shift 2 ;;
        --disk)      REQUEST_DISK="$2"; shift 2 ;;
        --env)       CONDA_ENV="$2"; shift 2 ;;
        --max-stall) MAX_STALL="$2"; shift 2 ;;
        --no-port-reclaim) PORT_RECLAIM=0; shift ;;
        -h|--help)   sed -n '2,25p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option '$1' (try --help)" >&2; exit 64 ;;
    esac
done

cd "$SCRIPT_DIR"
[[ -f "$RUN_SCRIPT" ]] || { echo "FATAL: $RUN_SCRIPT not found in $SCRIPT_DIR" >&2; exit 66; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$METADIR"
SUBFILE="$METADIR/torsiondrive.sub"
FLAGS_STR="--max-hours $MAX_HOURS --env $CONDA_ENV --max-stall $MAX_STALL"
[[ $PORT_RECLAIM -eq 0 ]] && FLAGS_STR="$FLAGS_STR --no-port-reclaim"

# ── Submit file ─────────────────────────────────────────────────────────────
# NMRbox has a shared filesystem, so no file transfer and the job runs in place.
cat > "$SUBFILE" <<EOF
# PCP1 TorsionDrive - generated by submit_condor.sh at $STAMP.
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

# Segmented running: the wrapper exits $REQUEUE_EXIT_CODE when its wall-clock
# budget expires with grid points left to do. Keeping such a job in the queue
# makes HTCondor run it again, so the scan proceeds as a chain of sub-8h
# segments. Any other exit code (0 = scan finished, anything else = a real
# failure) removes the job, so a broken run cannot loop forever.
on_exit_remove          = (ExitCode =!= $REQUEUE_EXIT_CODE)

# Record the machines this job has run on, so a retry can be steered away from
# one that just failed, e.g. by appending to a requirements line:
#   requirements = (TARGET.Machine =!= "lanthanum.nmrbox.org")
job_machine_attrs       = Machine
job_machine_attrs_history_length = 4

batch_name              = PCP1-torsiondrive-$STAMP

arguments               = --exec $FLAGS_STR
log                     = $METADIR/condor.log
output                  = $METADIR/condor.out
error                   = $METADIR/condor.err

queue 1
EOF

cat <<EOF

Submission : $STAMP
Resources  : $REQUEST_CPUS cpu, $REQUEST_MEMORY mem, $REQUEST_DISK disk, no GPU
Segments   : up to $MAX_HOURS h each, requeued until the scan finishes
Conda env  : $CONDA_ENV
Bookkeeping: $METADIR

EOF

if [[ $DRY_RUN -eq 1 ]]; then
    echo "--dry-run: not submitting. Submit file is at:"
    echo "  $SUBFILE"
    exit 0
fi

command -v condor_submit >/dev/null 2>&1 \
    || { echo "condor_submit not found - are you on a submit host?" >&2; exit 1; }

condor_submit "$SUBFILE" | tee "$METADIR/condor_submit.out"

cat <<EOF

Track:    condor_q -batch-name PCP1-torsiondrive-$STAMP
Segments: grep '^# === segment' $METADIR/job_info.txt
Progress: tail -f $SCRIPT_DIR/td_logs/torsiondrive.log | tr -d '\r'
Archive:  $METADIR/segments/<NNN>/   (per-segment copy of td_logs)
EOF
