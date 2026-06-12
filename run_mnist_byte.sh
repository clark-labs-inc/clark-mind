#!/usr/bin/env bash
set -e
python byte_multimodal_residual_memory.py --mode mnist_byte --train_limit 6000 --test_limit 1000 --epochs 2 --dim 256 --max_neurons 6000
