#!/bin/bash
# ------------------------------------------------------------------
# Run TorsionDrive + psi4 in parallel using cctools work_queue.
#
# Linux version (util-linux `script`). For macOS, use run_macos.sh.
#
# Usage:
#   ./run.sh
#
# Requires: torsiondrive, psi4, cctools (work_queue_worker) all on PATH
# ------------------------------------------------------------------

set -euo pipefail

# ---------------- User settings ----------------
QC_INPUT="input_pcm.dat"              # psi4 input template
DIHEDRALS="dihedrals.txt"     # dihedral definition file
GRID_SPACING=30               # degrees
WQ_PORT=50124                 # work_queue master port
WQ_PASSWORD_FILE="$(pwd)/wq.password"
WQ_SSL=1
export WQ_PASSWORD_FILE WQ_SSL

TOTAL_CORES=50
NUM_WORKERS=5                 # number of parallel work_queue_worker processes
THREADS_PER_WORKER=$((TOTAL_CORES / NUM_WORKERS))
# Default split: 4 workers x 4 threads = 16 cores.
# Adjust NUM_WORKERS (and THREADS_PER_WORKER updates automatically)
# e.g. NUM_WORKERS=8 -> 8 workers x 2 threads.

LOGDIR="td_logs"
PTY_COLS=250   # wide enough that torsiondrive's colored output never wraps mid-line
# ------------------------------------------------

mkdir -p "$LOGDIR"

if (( NUM_WORKERS * THREADS_PER_WORKER > TOTAL_CORES )); then
    echo "ERROR: workers x threads-per-worker exceeds TOTAL_CORES. Fix the numbers above." >&2
    exit 1
fi

echo "Launching TorsionDrive master on port ${WQ_PORT} ..."
# NOTE: we deliberately avoid "cmd | tee log &" here, because in bash
# $! after a backgrounded pipeline captures the LAST command's PID (tee),
# not the actual torsiondrive-launch/script process. Using process
# substitution instead means $! correctly refers to the real master
# process, so cleanup() can actually find and kill it.
#
# We also wrap the command in `script` with an explicit pty width so
# (a) colored/progress output survives being non-interactive under tee,
# and (b) the pty doesn't wrap lines at some default/stale width once
# backgrounded (backgrounded jobs don't get SIGWINCH resize updates).
#
# NOTE: this is the Linux (util-linux) `script` calling convention:
#   script -q -c "command string" /dev/null
# It only accepts a single command string via -c (not a program + argv
# list like BSD/macOS script), so we build that string with bash's own
# quoting instead of passing positional params. See run_macos.sh for
# the macOS version of this script.
TD_CMD=$(printf 'stty cols %q rows 50 2>/dev/null; exec torsiondrive-launch %q %q -g %q -e psi4 --wq_port %q -v' \
    "$PTY_COLS" "$QC_INPUT" "$DIHEDRALS" "$GRID_SPACING" "$WQ_PORT")
script -q -c "$TD_CMD" /dev/null \
    > >(tee "${LOGDIR}/torsiondrive.log") 2>&1 &
TD_PID=$!
echo "TorsionDrive master PID: $TD_PID"

# Give the master a moment to open the port before workers connect
sleep 3

echo "Starting ${NUM_WORKERS} work_queue_worker(s), ${THREADS_PER_WORKER} thread(s) each ..."
WORKER_SSL_FLAG=""
[[ -n "${WQ_SSL:-}" ]] && WORKER_SSL_FLAG="--ssl"
WORKER_PIDS=()
for i in $(seq 1 "$NUM_WORKERS"); do
    PSI4_NUM_THREADS="$THREADS_PER_WORKER" \
        work_queue_worker localhost "$WQ_PORT" \
        --cores "$THREADS_PER_WORKER" \
	--memory 48000 \
	--disk 80000 \
	-P "$WQ_PASSWORD_FILE" \
	$WORKER_SSL_FLAG \
        > "${LOGDIR}/worker_${i}.log" 2>&1 &
    WORKER_PIDS+=($!)
    LAST_IDX=$((${#WORKER_PIDS[@]} - 1))
    echo "  worker $i PID: ${WORKER_PIDS[$LAST_IDX]}"
done

# ---------------- Cleanup on exit / Ctrl-C ----------------
cleanup() {
    echo ""
    echo "Stopping workers and master ..."
    kill "${WORKER_PIDS[@]}" 2>/dev/null || true
    kill "$TD_PID" 2>/dev/null || true
    # `script` runs torsiondrive-launch as a child inside its own pty/session,
    # so killing $TD_PID (the `script` process) does not reliably kill that
    # child. Clean it up explicitly by name as a safety net.
    pkill -f "torsiondrive-launch" 2>/dev/null || true
    pkill -f "work_queue_worker" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
# -----------------------------------------------------------

echo ""
echo "TorsionDrive is running. Tail progress with:"
echo "  tail -f ${LOGDIR}/torsiondrive.log | tr -d '\\r'"
echo "Worker logs are in ${LOGDIR}/worker_*.log"
echo ""
echo "Waiting for TorsionDrive master to finish (Ctrl-C to stop everything early) ..."

wait "$TD_PID"
echo "TorsionDrive master has finished."
