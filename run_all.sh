#!/usr/bin/env bash
set -euo pipefail

STATES=(holo cys-loaded)
WATERS=(tip3p opc)
FFS=(ff14sb ff19sb)
SEEDS=(0 1 2)

for state in "${STATES[@]}"; do
    for ff in "${FFS[@]}"; do
        for water in "${WATERS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "=== state=$state ff=$ff water=$water seed=$seed ==="
                python simulate.py \
                    --state "$state" \
                    --ff    "$ff"    \
                    --water "$water" \
                    --seed  "$seed"
            done
        done
    done
done
