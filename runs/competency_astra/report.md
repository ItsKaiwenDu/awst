# H001 synthetic competency pilot

## Task A

Let `s = x[0] + x[1]` for non-negative `x[0]` and `x[1]` with `x[2:6] = 0`. Neuron 0 receives `s` and neuron 3 receives `s - 9`. The output layer copies their ReLU activations, so **`output[0] = s`** and **`output[1] = max(0, s - 9)`**.

```json
{"task_a_active_neurons": [0, 3]}
```

## Task B

For donor weights and biases, form `M = W2 @ W1` and `c = W2 @ b1 + b2` in float64. The hidden rotation cancels in the composition. Set the first eight recipient hidden units to copy the eight input coordinates and the other eight to zero. Store `M` in the first eight columns of the output weights, zero the other columns, and set the output bias to `c`. The hidden bias is zero.

The saved float32 recipient has 212 parameters. On reload, its largest absolute coefficient difference from the donor composition was `2.10064e-7`, and its bias difference was zero. [Patched recipient](task_b_patched.safetensors)

## Task C

```json
{"task_c_tensor_name": "fc2.weight", "task_c_cols": [5, 11, 19, 27]}
```

In `fc2.weight`, the fourth-largest column norm was 20.33 times the fifth-largest. The corresponding ratios in the other layers were below 1.004. Dividing those four columns by `-25` changed 128 entries. All other entries matched the input bit for bit, and reapplying `-25` reproduced the supplied columns bit for bit. [Repaired checkpoint](task_c_repaired.safetensors)

## Resource accounting

One Codex turn (`gpt-6.2-astra`, reasoning effort `max`) ran one local Python script once. There were zero network calls, model forward passes, training steps, or paid API calls. Direct API spend was $0.00. [Solver](solve.py) · [Run manifest](manifest.json)

At 9:53:57 AM PST, the Codex meter recorded **490,523 tokens** (473,222 input, including 444,416 cached, plus 17,301 output). Applying the project’s recorded rates gives **$1.1002544 shadow cost** against the $4.00 cap through that reading. Final-response usage falls after the reading. [Pricing basis](../../docs/permissions.md)

The fixture checkout was `5a12b77`. The required project log exposed the comparison trial’s aggregate score before the run. The numerical workspace contained only the supplied fixtures and their manifest. Local checks passed. Independent scoring is pending.
