# Design: Growing Residual Routing Memory

## Goal

Design a learning system that does not use backpropagation. The system learns from residuals, dynamically adds neurons, chooses better nearby neurons, chooses better communication operations, forgets weak neurons, and supports bounded recursive/self-loop behavior.

## Neuron structure

A neuron is a local expert:

```text
neuron_i = {
    center c_i,              # feature/state prototype
    vote v_i,                # class-message vector
    sigma_i,                 # receptive-field radius
    reliability r_i,         # recent usefulness
    usage_i,                 # activity trace
    op_q_i,                  # operation preference scores
    age_i                    # local lifetime counter
}
```

In the MNIST prototype, `vote` is a 10-dimensional class vector.

## Prediction

For an input feature vector `z`, the system performs bounded recursive routing:

```text
state_0 = z
logits_0 = 0

for step in 1..T:
    choose k nearby neurons to current state
    compute activation weights
    each selected neuron sends a class message
    add weighted messages into logits
    move state slightly toward the selected local manifold
    apply repeat penalty so one neuron cannot dominate forever

prediction = argmax(logits)
```

This creates a small active path through a large memory. Cost depends on active neurons, not total neurons.

## Residual update

For classification:

```text
p = softmax(logits)
target = one_hot(label), with slight negative mass on non-target classes
residual = target - p
```

Only neurons that participated in the route are updated:

```text
v_i <- v_i + learning_rate * activation_i * residual
```

This is local residual correction, not backpropagation.

## Choosing better nearby neurons

The first prototype uses Euclidean nearest neighbors in feature space plus reliability and repeat penalties. The scalable design should use a richer routing score:

```text
score_i =
    distance_to_center_i
  - reliability_bonus_i
  - learned_edge_bonus_i
  + repeat_penalty_i
  + energy_cost_i
```

Future versions can add:

```text
local learned metrics
coarse hash routing
approximate nearest neighbor indexes
graph-edge routing
residual-specialist routing
curiosity/exploration routing
```

## Operation choice

Each neuron has several possible communication operations. The starter code includes three:

```text
op 0: normal class message
op 1: sharper locality-gated message
op 2: reliability-amplified message
```

Each neuron stores `op_q[i, op]`. During training, it sometimes explores; otherwise, it chooses its best operation.

A selected operation is rewarded when its message aligns with the residual:

```text
reward = dot(v_i, residual) * activation_i
op_q[i, op] <- moving_average(op_q[i, op], reward)
```

This is closer to a local bandit rule than gradient descent.

## Dynamic growth

A new neuron is added when the current system cannot explain the sample well:

```text
if true_class_probability is low
or nearest_neuron_distance is high:
    add neuron centered at current feature
```

The new neuron receives an initial class message based on the residual and the true class.

A larger-scale growth rule should be stricter:

```text
grow_score =
    residual_size
  + novelty
  + repeated_unresolved_error
  - redundancy
  - tile_pressure
```

Grow only when unresolved errors repeat, not for every random mistake.

## Forgetting

Forgetting is necessary. The starter code prunes when capacity is exceeded using reliability, usage, and vote strength.

More advanced forgetting should have multiple stages:

```text
soft forget: reduce reliability
shrink: reduce radius
mute: suppress vote
merge: combine with a similar neuron
delete: remove fully
```

Useful rare neurons should be protected. Useless, harmful, redundant, or over-broad neurons should fade.

## Recursive/self-loop behavior

Recursive behavior is implemented by allowing multiple routing steps over a changing state. A neuron can be selected again, but repeated use is penalized.

Safety controls:

```text
maximum routing steps
repeat penalty
state normalization
message norm control
energy budget
stop when confidence stops improving
```

The guiding rule is:

```text
A recursive loop should survive only if it improves prediction or uncertainty.
```
