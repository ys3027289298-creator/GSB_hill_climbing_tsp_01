# Hill_Climbing_TSP

This is a simulation of the Hill Climbing Algorithm (Artificial
Intelligence) in Python. The simulation depicts the entire state space
search according to the algorithm, i.e. it shows traversing down the
nodes as per their heuristic value.

## Usage

```
python3 newmodified.py [input_file] [--max-iterations N]
                       [--seed S] [--restarts R] [--plot]
```

* `input_file` defaults to `input.txt`. Format: first line `n`, then a
  line of `n` x coordinates, then a line of `n` y coordinates.
* `--max-iterations` caps the number of accepted swaps per run
  (default 200).
* `--seed` / `--restarts` add `R` random-restart starts drawn from a
  local `random.Random(seed)`; equal seeds give bitwise-identical output.
* `--plot` animates the search with matplotlib (off by default so the
  solver runs headless).

Console fields are the historical ones: `iteration =`,
`total distance =`, `swapped`, `finish`, `Final order =`,
`Initial cost =`, `Final cost =`. The final non-improving neighbourhood
scan is no longer printed as an extra iteration; every printed iteration
corresponds to exactly one accepted swap.

## Invariants

The solver maintains three runtime-verified invariants (see the module
docstring of `newmodified.py`):

1. **COST** — the running cost always equals the exact Euclidean tour
   cost of the current order; `swap_delta` is algebraically exact for
   every `n >= 2` (adjacent swaps, the wrap-around pair, and the
   degenerate 2-city tour included), and the equality is asserted after
   every accepted swap.
2. **ACCEPT** — a swap is applied iff its exact delta is strictly
   negative, so the tour cost strictly decreases on every iteration.
3. **TERM** — the loop stops only when no strictly-improving swap exists
   (strict 2-swap local optimum) or the iteration budget is exhausted.

Malformed input (empty file, non-positive `n`, missing/extra
coordinates, non-numeric or non-finite values) exits with status 2 and a
clear `error:` message on stderr instead of silently returning a wrong
result.

## Tests

```
python3 -m unittest test_newmodified -v
```

The suite checks bitwise-identical output for a fixed seed, compares
small instances against the exhaustive optimum, exercises the edge cases
(1/2 cities, duplicate coordinates, no extra stop iteration) and
verifies that invalid input fails loudly.
