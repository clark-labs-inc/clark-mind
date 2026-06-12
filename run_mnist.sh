#!/usr/bin/env bash
set -euo pipefail
python growing_residual_mnist.py --train_limit 12000 --test_limit 2000 --epochs 2 --max_neurons 4000
