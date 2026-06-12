# Growing Residual Learning: No-Backprop Prototypes

This package contains clean-slate prototypes for learning systems that avoid backpropagation. They use local residual correction, dynamically added neurons, routing among nearby neurons, local operation selection, forgetting/pruning, and bounded recursive loops.

There are two tracks:

```text
1. growing_residual_mnist.py
   Original no-backprop MNIST prototype using a fixed unsupervised image feature encoder.

2. byte_multimodal_residual_memory.py
   Byte/event-level multimodal prototype where text, image, audio, metadata, queries,
   and targets are all represented in one byte-event language.
```

There is no `loss.backward()`, no optimizer, and no learned layered encoder in these prototypes.

## Core idea

Instead of layers trained by gradient descent, the model is a growing graph/memory of local experts:

```text
input stream
 -> fixed non-backprop encoder
 -> find nearby useful neurons
 -> neurons send output-message vectors
 -> bounded recursive routing updates state/logits
 -> prediction
 -> residual = target - prediction
 -> update only participating neurons
 -> add/forget neurons when needed
```

## Byte/event-level multimodal version

The multimodal version treats every input as byte events:

```text
event = {
    modality,        # text, image, audio, metadata, etc.
    role,            # observed input, query, target, context
    channel,         # grayscale channel, audio channel, text stream, etc.
    position/time,   # x/y for image, t for audio/text
    byte_value       # 0..255
}
```

Examples:

```text
text  -> UTF-8 bytes
image -> uint8 pixel bytes + row/column/channel tags
audio -> uint8/PCM bytes + time/channel tags
metadata/query/control -> ordinary byte events with role tags
```

A fixed hash/sketch encoder maps these events into a shared state space. The growing residual memory then learns from local residuals exactly as before.

The output is byte-level too. For MNIST-style digit classification, the answer is the ASCII byte `'0'..'9'`, not a special neural-network class head.

## Files

```text
growing_residual_mnist.py              Runnable no-backprop MNIST prototype
byte_multimodal_residual_memory.py     Runnable byte/event-level multimodal prototype
DESIGN.md                              Main local residual learning design
MULTIMODAL_BYTE_DESIGN.md              Byte/multimodal architecture notes
SCALING.md                             Notes on sparse brain-like scaling
EXPERIMENTS.md                         Suggested experiments and ablations
requirements.txt                       Python dependencies
run_mnist.sh                           Example original MNIST launcher
run_byte_multimodal_toy.sh             Example multimodal toy launcher
run_mnist_byte.sh                      Example MNIST-as-byte-events launcher
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The built-in byte/event multimodal toy task only needs NumPy. The MNIST runs use PyTorch/torchvision for loading MNIST, and the original MNIST prototype also uses scikit-learn for PCA.

## Run original MNIST prototype

```bash
python growing_residual_mnist.py --train_limit 12000 --test_limit 2000 --epochs 2 --max_neurons 4000
```

Or:

```bash
bash run_mnist.sh
```

## Run byte/event-level multimodal toy task

This generated task mixes image bytes, audio bytes, text query bytes, and metadata bytes. The target is an ASCII digit byte.

```bash
python byte_multimodal_residual_memory.py --mode toy_multimodal --train_limit 6000 --test_limit 1000 --epochs 2
```

Or:

```bash
bash run_byte_multimodal_toy.sh
```

## Run MNIST as byte/event streams

This treats MNIST images as byte/event streams, adds text query bytes, and predicts an ASCII digit byte.

```bash
python byte_multimodal_residual_memory.py --mode mnist_byte --train_limit 6000 --test_limit 1000 --epochs 2
```

Or:

```bash
bash run_mnist_byte.sh
```

## What to look for

The first useful questions are:

```text
1. Does dynamic growth improve accuracy over fixed memory?
2. Does recursive routing improve accuracy or only add instability?
3. Do operation scores specialize?
4. Does forgetting remove harmful/redundant neurons?
5. Does byte/event representation allow image, text, audio, and metadata to share one memory?
6. Does performance keep improving as neurons are added, or does it memorize?
```

This is a research prototype. It is deliberately small, readable, and easy to modify.
