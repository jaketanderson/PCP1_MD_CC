#!/usr/bin/env bash
set -euo pipefail

STATES=(holo cys-loaded)
WATERS=(tip3p opc)
FFS=(ff14sb ff19sb)
SEEDS=(0 1 2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/slurm_logs"
mkdir -p "${LOGS_DIR}"

for state in "${STATES[@]}"; do
    for ff in "${FFS[@]}"; do
        for water in "${WATERS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                job="${state}_${ff}_${water}_${seed}"
                echo "Submitting: ${job}"
                sbatch \
                    --job-name="${job}" \
                    --partition=gpu \
                    --gpus=1 \
                    --mem=4G \
                    --time=5-00:00:00 \
                    --output="${LOGS_DIR}/${job}.out" \
                    --error="${LOGS_DIR}/${job}.err" \
                    --wrap="cd ${SCRIPT_DIR} && conda run -n openmm2 python simulate.py \
                        --state ${state} \
                        --ff    ${ff}    \
                        --water ${water} \
                        --seed  ${seed}"
            done
        done
    done
done
