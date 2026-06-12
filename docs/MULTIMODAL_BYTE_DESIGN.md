# Byte/Event-Level Multimodal Design

## Clean-slate change

The original MNIST prototype used a fixed image feature vector and a 10-class vote. The byte/event version removes that assumption.

Instead of saying:

```text
image -> image encoder -> class logits
text  -> text encoder  -> token logits
audio -> audio encoder -> label logits
```

we say:

```text
observation -> byte/event stream -> shared residual memory -> byte messages
```

Everything is represented as small events:

```text
event = {
    modality,        # text, image, audio, metadata, etc.
    role,            # observed input, query, target, context
    channel,         # grayscale channel, audio channel, text stream, etc.
    position/time,   # x/y for image, t for audio/text
    byte_value       # 0..255
}
```

This makes the system naturally byte-level while still preserving enough structure for different modalities.

## Why not pure raw byte strings only?

A totally flat byte string is possible, but it throws away useful spatial and temporal identity. Pixel byte `65`, text byte `65` (`A`), and audio amplitude byte `65` should not automatically mean the same thing.

So the base unit is not just:

```text
byte
```

It is:

```text
byte + where it came from + what role it plays
```

The system can still learn cross-modal bridges because all events enter the same memory and the same residual update rule.

## Shared output language

The output is also byte-level. For MNIST-style digit classification, the answer is the ASCII byte:

```text
'0'..'9'  ->  48..57
```

For future tasks, outputs can be ordinary UTF-8 bytes, audio bytes, image bytes, or structured event bytes.

That means the same memory can be trained to do tasks like:

```text
image bytes + query bytes -> answer byte/text bytes
text bytes -> next text byte
image bytes -> missing patch bytes
audio bytes -> text bytes
image + text + audio -> shared answer bytes
```

## Residual rule

The residual remains local:

```text
prediction = byte distribution for requested output slot
target     = target byte distribution
residual   = target - prediction
```

Only neurons that participated in the route update their byte-message vector:

```text
message_i <- message_i + learning_rate * activation_i * residual
```

There is still no backward pass and no optimizer.

## Universal byte/event sketch

The included `ByteEventSketcher` is a fixed feature-hashing encoder. It is not learned. It maps event streams into vectors by hashing fields such as:

```text
(modality, role, channel, position/time, byte)
(modality, role, channel, byte)
(modality, role, previous_byte, current_byte)
```

This gives the residual memory a common searchable space without a trained neural encoder.

## Multimodal routing

A neuron can become local to any recurring mixed pattern:

```text
image stroke + text query pattern
image region + audio rhythm
audio byte pattern + answer byte
text word + visual shape
metadata context + all of the above
```

Routing still works the same way:

```text
current event sketch
 -> find nearby useful neurons
 -> neurons emit byte messages
 -> recursive state update
 -> answer byte distribution
 -> residual update active neurons
```

## Dynamic growth

A new neuron is added when the current active path cannot explain the target byte:

```text
if target_byte_probability is low
or nearest_event_pattern is far:
    add a new byte/event neuron
```

In a larger system, growth should require repeated unresolved residuals, not one random mistake.

## Forgetting

Forgetting becomes even more important in multimodal byte space because the input space is huge. Weak neurons should be softened, merged, or removed when they are:

```text
rarely used
locally harmful
duplicated by nearby neurons
too broad / confused across output bytes
obsolete after better neurons appear
```

## Recursive loops

Recursive/self-loop behavior is still bounded. A neuron can fire again on later routing steps, but repeat penalties and finite energy prevent runaway loops.

Useful loops should become local attractors:

```text
ambiguous event pattern -> clarify state -> better nearby neurons -> better byte prediction
```

Harmful loops should lose reliability.

## Included demo script

`byte_multimodal_residual_memory.py` has two modes:

```bash
python byte_multimodal_residual_memory.py --mode toy_multimodal
python byte_multimodal_residual_memory.py --mode mnist_byte --train_limit 6000 --test_limit 1000
```

`toy_multimodal` uses generated image bytes, audio bytes, text query bytes, and metadata bytes. The target is a digit byte.

`mnist_byte` treats MNIST images as byte/event streams, adds text query bytes, and predicts an ASCII digit byte.

## What this buys us

The architecture becomes:

```text
one memory
one event language
one residual rule
many modalities
byte-level input
byte-level output
local growth
local forgetting
bounded recursion
no backpropagation
```

This is much closer to the clean-slate goal than a special-purpose image classifier.
