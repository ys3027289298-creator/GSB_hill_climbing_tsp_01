# Hill_Climbing_TSP

This is a simulation of Hill Climbing Algorithm (Artificial Intelligence) in Python.The simulation depicts entire state space search according to algorithm, i.e. it shows traversing down the nodes as per their heuristic value.

## Usage

```
python3 newmodified.py [input.txt] [--seed N] [--shuffle] [--max-iterations N] [--no-plot]
```

- Invalid input (empty city list, wrong coordinate counts, non-numeric or
  non-finite values) exits with status 2 and an explicit `error:` message.
- Two-city and single-city tours are handled correctly (length `2*d` and `0`),
  and identical coordinates terminate immediately with cost `0`.
- `--shuffle` uses an isolated `random.Random(seed)`; the same seed reproduces
  byte-identical output.
- Acceptance invariant: only a strictly improving 2-swap is accepted and the
  cost is recomputed exactly after every move, so termination is guaranteed
  and `finish` certifies a 2-swap local optimum.
- Run the tests with `python3 -m unittest test_newmodified`.
