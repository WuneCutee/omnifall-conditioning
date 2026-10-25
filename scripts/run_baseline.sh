#!/bin/bash
# Bước 1: Tái lập baseline paper OmniFall
set -e

python -m src.train --config configs/default.yaml --model baseline
python -m src.evaluate --config configs/default.yaml \
    --checkpoint checkpoints/baseline_best.pt --model-type baseline
