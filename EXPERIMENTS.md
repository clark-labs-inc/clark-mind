# Suggested Experiments and Ablations

## Original MNIST smoke run

```bash
python growing_residual_mnist.py --train_limit 2000 --test_limit 500 --epochs 1 --max_neurons 1000
```

## Original MNIST starter run

```bash
python growing_residual_mnist.py --train_limit 12000 --test_limit 2000 --epochs 2 --max_neurons 4000
```

## Byte/event multimodal toy smoke run

This uses generated image bytes, audio bytes, text query bytes, metadata bytes, and an ASCII digit-byte target.

```bash
python byte_multimodal_residual_memory.py --mode toy_multimodal --train_limit 1000 --test_limit 300 --epochs 1 --max_neurons 1200
```

## Byte/event MNIST starter run

```bash
python byte_multimodal_residual_memory.py --mode mnist_byte --train_limit 6000 --test_limit 1000 --epochs 2 --max_neurons 6000
```

## Recursion ablation

Compare:

```bash
python byte_multimodal_residual_memory.py --mode toy_multimodal --steps 1 --epochs 2
python byte_multimodal_residual_memory.py --mode toy_multimodal --steps 3 --epochs 2
python byte_multimodal_residual_memory.py --mode toy_multimodal --steps 5 --epochs 2
```

Questions:

```text
Does recursion improve test accuracy?
Does recursion increase overfitting?
Does it stabilize or destabilize routing?
Do repeated neurons become useful local loops or harmful attractors?
```

## Neighborhood size ablation

Compare:

```bash
python byte_multimodal_residual_memory.py --mode toy_multimodal --k 8  --epochs 2
python byte_multimodal_residual_memory.py --mode toy_multimodal --k 24 --epochs 2
python byte_multimodal_residual_memory.py --mode toy_multimodal --k 64 --epochs 2
```

Questions:

```text
Does a larger neighborhood help because more neurons vote?
Does it hurt because unrelated byte/event neurons interfere?
```

## Capacity ablation

Compare:

```bash
python byte_multimodal_residual_memory.py --mode toy_multimodal --max_neurons 1000  --epochs 2
python byte_multimodal_residual_memory.py --mode toy_multimodal --max_neurons 6000  --epochs 2
python byte_multimodal_residual_memory.py --mode toy_multimodal --max_neurons 12000 --epochs 2
```

Questions:

```text
Does more memory improve generalization?
When does it become memorization?
Does pruning remove confused or redundant neurons?
```

## Byte sketch dimension ablation

Compare:

```bash
python byte_multimodal_residual_memory.py --mode toy_multimodal --dim 64  --epochs 2
python byte_multimodal_residual_memory.py --mode toy_multimodal --dim 256 --epochs 2
python byte_multimodal_residual_memory.py --mode toy_multimodal --dim 512 --epochs 2
```

Questions:

```text
Does a richer byte/event sketch help routing?
Does higher dimension make nearest-neighbor search more sparse or less stable?
```

## MNIST byte threshold ablation

Compare sparse nonzero-ish pixel events against all pixel bytes:

```bash
python byte_multimodal_residual_memory.py --mode mnist_byte --image_threshold 8  --epochs 2
python byte_multimodal_residual_memory.py --mode mnist_byte --image_threshold 32 --epochs 2
python byte_multimodal_residual_memory.py --mode mnist_byte --include_zero_pixels --epochs 2
```

Questions:

```text
Are zero/background bytes useful context or mostly noise?
Does thresholding act like a helpful non-learned attention mechanism?
```

## Metrics to add next

The scripts currently print online accuracy, test accuracy, and neuron count. Useful next metrics:

```text
average residual norm
neurons added per epoch
neurons pruned per epoch
operation usage frequencies
operation reward averages
byte confusion matrix
active-path entropy
repeat-loop frequency
per-output-byte neuron counts
modality-drop robustness
```
