# Scaling Notes: Toward Brain-Like Sparse Systems

## Direct brain-size scaling is not the right first step

A human brain is often described as having tens of billions of neurons and vastly more synapses. Copying that literally is not useful for MNIST. MNIST has only 60,000 training images and 10,000 test images, so billions of neurons would mostly create extreme overcapacity.

For MNIST, useful experimental ranges are more like:

```text
1,000 neurons       tiny proof of concept
10,000 neurons      interesting
100,000 neurons     probably plenty
1,000,000 neurons   already excessive for MNIST
```

The goal is to prove the learning rule and dynamics first.

## The only scalable form: huge memory, tiny active path

The system must never search or update every neuron.

Wrong:

```text
for every input:
    compare to every neuron
    update many global weights
```

Right:

```text
for every input:
    choose a few regions/tiles
    search a small local index
    activate a tiny number of neurons
    update only the active path
```

Cost should scale as:

```text
routing_steps * active_neurons * feature_size
```

not:

```text
total_neurons * feature_size
```

## Regional/tiled architecture

A scalable version should be organized as local neighborhoods:

```text
system
  region_0
    tile_0
      neurons
      local index
      local edges
      local growth budget
      local forgetting policy
    tile_1
    tile_2
  region_1
  region_2
```

Each tile is responsible for local routing, local neuron birth/death, and local residual statistics.

## Sparse communication graph

Every neuron should have a small number of learned edges:

```text
short-range edges: local neighbors and frequent co-activations
long-range edges: rare, high-value communication paths
self-edges: recursive attractors, energy-bounded
```

Edges should be strengthened when communication reduces residual and weakened when it increases confusion.

## Growth at scale

At large scale, growth must be budgeted:

```text
if error is high
and novelty is high
and the same error repeats
and tile has capacity:
    grow neuron
else:
    adjust existing neurons or route differently
```

Uncontrolled growth becomes memorization.

## Forgetting at scale

Forgetting should run continuously and locally. Each tile should ask:

```text
Which neurons are useful?
Which are duplicate?
Which are harmful?
Which are stale?
Which are too broad/confused?
```

Use soft forgetting before deletion. Merge similar neurons when possible.

## Recursion at scale

Recursive loops should use an energy budget:

```text
energy = fixed_budget
while energy remains:
    route
    communicate
    update state
    stop if confidence improves no further
```

Self-loops are allowed, but they must pay energy and earn reliability by improving outcomes.

## Staged path

```text
Stage 1: MNIST
10k-100k neurons
prove residual learning, growth, forgetting, recursion

Stage 2: harder vision
1M-10M neurons
test routing and forgetting under real diversity

Stage 3: multimodal memory
100M-1B neurons
distributed tiles, sparse activation, approximate routing

Stage 4: brain-scale research system
10B-100B neurons
cluster/neuromorphic implementation, strict locality, sleep/merge/prune cycles
```

The key principle is ecological scaling: neurons are born, specialize, communicate, compete, sleep, merge, forget, and sometimes form recursive loops.
