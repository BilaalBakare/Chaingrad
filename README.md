# chaingrad

A scalar-valued automatic differentiation engine, built from scratch to understand how backpropagation actually works under the hood — the same reverse-mode autodiff pattern used by PyTorch's `autograd`, implemented at small scale and explained at every step.

## Demo

```python
from chaingrad.engine import Value

def f(x):
    return 3*(x**2) + x + 5

a = Value(6)
result = f(a)
result.backward()

print(result.data)   # 119
print(a.grad)         # 37.0
```

**What's happening:** `f(x) = 3x² + x + 5`. Evaluating `f(6)` gives `119` — that's just the forward pass, ordinary arithmetic. But because every operation (`**`, `*`, `+`) on a `Value` silently records how it was computed, calling `result.backward()` walks that hidden record backward and computes `df/dx` automatically: `df/dx = 6x + 1`, which at `x = 6` is `37.0` — exactly what `a.grad` reports, with no manual differentiation written anywhere.

That's the entire idea behind this library: wrap numbers so they remember their own history, and gradients fall out for free.

## Why this exists

Most autodiff tutorials either hand you finished code to copy, or stay purely theoretical and never touch an implementation. This project is neither — it's a from-scratch build, done one operation at a time, with the reasoning behind each design decision worked out before any code was written: why reverse-mode autodiff (not numerical or symbolic differentiation) is what deep learning uses, why each operation owns its own local derivative via a closure instead of one big dispatch function, and why topological sort — not just any traversal order — is required to get correct gradients out of a graph where a value can be reused multiple times.

## How it works, briefly

1. **`Value`** wraps a plain number and adds bookkeeping: `data` (the value itself), `grad` (accumulates `∂output/∂this`, filled in only after `.backward()` runs), `_prev` (the parent `Value`s that produced this one), and `_backward` (a closure holding the local derivative rule for whichever operation created it).
2. **Operator overloading** (`__add__`, `__mul__`, `__pow__`, ...) builds a computational graph automatically, just by using `+`, `*`, `**` normally — no manual graph construction needed.
3. **`.backward()`** performs a post-order depth-first topological sort starting from the output node, seeds `self.grad = 1.0`, then walks the reversed order calling each node's `_backward` closure — guaranteeing every node's gradient is fully accumulated before anything downstream tries to read it.

## Status

- [x] Core `Value` class with graph tracking (`_prev`, `_op`, `_backward`)
- [x] `+`, `*`, `**`, unary `-` — derived and gradient-checked against numerical differentiation
- [x] `-`, `/` and their reflected counterparts (`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`) — composed from the above
- [x] `.backward()` — topological sort + reverse traversal
- [ ] `nn.py` — `Neuron`, `Layer`, `MLP` built on top of `Value`
- [ ] Training example on a toy dataset
- [ ] Packaged as an editable install (`pyproject.toml`)
- [ ] Tensor-level autograd (broadcasting, matmul) — deliberately out of scope for now; scalar autograd first

## Project layout

```
Chaingrad/
├── chaingrad/
│   ├── __init__.py
│   └── engine.py     # Value class — the autodiff engine itself
├── testing/
│   └── test.ipynb    # exploratory / gradient-check notebook
├── .gitignore
└── README.md
```

## Running it locally

For now, notebooks import the package via a relative path hack rather than an installed package:

```python
import sys
sys.path.append('..')
from chaingrad.engine import Value
```

This works because `testing/` sits one level below the project root, so `..` puts `chaingrad/` on the import path. Once `nn.py` and an `examples/` folder are added, this project will move to a proper editable install (`pip install -e .` via `pyproject.toml`) so imports work the same way from anywhere in the project.
