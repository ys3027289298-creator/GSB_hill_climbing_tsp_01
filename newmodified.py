"""Hill-climbing TSP solver.

Reads a TSP instance (number of cities, x coordinates, y coordinates)
and improves the tour with a steepest-descent 2-swap (order inversion)
hill climb.

Algorithm invariants
---------------------
The implementation carries three explicit, checkable invariants.  They
are documented on :func:`hill_climb` and verified at runtime, so the
acceptance and termination behaviour is provable rather than incidental:

1. COST    -- at every loop boundary ``current_cost`` equals the exact
              Euclidean tour cost of the current order.  A swap changes
              the cost by :func:`swap_delta`, whose value is algebraically
              exact for every ``n >= 2`` (including adjacent swaps, the
              wrap-around pair (0, n-1) and the degenerate n == 2 case);
              the equality is re-derived by a full cost sum and asserted
              after every applied swap.
2. ACCEPT  -- a swap is applied if and only if its exact delta is
              strictly negative (``delta < 0``), and the minimum over all
              pairs is chosen.  Therefore the tour cost strictly decreases
              at every accepted iteration.
3. TERM    -- the loop terminates either because no strictly-improving
              swap exists (the current order is a strict local optimum
              in the 2-swap neighbourhood) or because the finite
              ``max_iterations`` budget is exhausted.  The number of
              accepted iterations is bounded by the budget, and on an
              early exit the local-optimality condition is guaranteed.

The historical per-iteration console fields (``iteration =``,
``total distance =``, ``swapped``, ``finish``, ``Final order =``,
``Initial cost =``, ``Final cost =``) are unchanged.  Unlike the old
script, the final non-improving neighbourhood scan is no longer printed
as an extra "iteration": the log contains exactly the accepted swaps and
is followed by a single ``finish`` line.
"""

import argparse
import math
import random
import sys


def distance(xcord, ycord, i, j):
    """Euclidean distance between cities i and j; always non-negative."""
    dx = xcord[i] - xcord[j]
    dy = ycord[i] - ycord[j]
    return math.sqrt(dx * dx + dy * dy)


def tour_cost(xcord, ycord, order):
    """Exact closed-tour cost (last city returns to the first).

    Defined as 0.0 for the trivial tours n <= 1; for n == 2 the single
    edge is traversed once in each direction, so the cost is 2 * d(0, 1).
    """
    n = len(order)
    total = 0.0
    for k in range(n):
        total += distance(xcord, ycord, order[k], order[(k + 1) % n])
    return total


def swap_delta(xcord, ycord, order, i, j):
    """Exact change in tour cost caused by swapping positions i and j.

    Only the undirected edges incident to position i or j can change;
    edges are deduplicated as unordered pairs so the formula is exact even
    when they coincide:

    * adjacent positions (including the wrap-around pair (0, n - 1))
      share the edge (i, j), which leaves and re-enters identically;
    * with n == 2 every incident edge is the same edge (0, 1), so the
      old code's four-term delta double-counted it and produced
      negative costs.  Deduplication makes the delta exactly 0.

    Edge pairs are summed in a fixed sorted order so the result is
    independent of hash-seed randomisation and bitwise reproducible.
    """
    n = len(order)
    edges = set()
    for p in (i, j):
        edges.add(tuple(sorted((p, (p - 1) % n))))
        edges.add(tuple(sorted((p, (p + 1) % n))))

    before = 0.0
    after = 0.0
    for a, b in sorted(edges):
        before += distance(xcord, ycord, order[a], order[b])
        moved_a = j if a == i else i if a == j else a
        moved_b = j if b == i else i if b == j else b
        after += distance(xcord, ycord, order[moved_a], order[moved_b])
    return after - before


def hill_climb(xcord, ycord, order, max_iterations, log=None):
    """Improve ``order`` in place by steepest-descent 2-swap hill climb.

    Returns ``(initial_cost, final_cost, accepted_iterations)``.

    If ``log`` is a list, one tuple
    ``(iteration, cost_before_swap, swap_i, swap_j)`` is appended for each
    accepted swap, which lets callers render exactly the historical console
    output deterministically.

    Invariants (see module docstring), verified at runtime:

    * ACCEPT: ``best_delta < 0`` is necessary and sufficient to apply a
      swap, hence ``current_cost`` strictly decreases on every iteration.
    * COST:   after applying the swap, the incrementally updated cost
      equals a freshly recomputed full tour cost.
    * TERM:   the while loop has a hard ``max_iterations`` bound and exits
      early only when no pair has a strictly negative delta, i.e. the
      order is a strict 2-swap local optimum.
    """
    n = len(order)
    current_cost = tour_cost(xcord, ycord, order)
    initial_cost = current_cost
    accepted = 0

    while accepted < max_iterations:
        best_delta = 0.0
        best_i = -1
        best_j = -1

        # Fixed scan order (i < j) => deterministic tie handling: the
        # first most-improving pair wins; equal deltas never replace it
        # because the comparison is strictly negative.
        for i in range(n - 1):
            for j in range(i + 1, n):
                delta = swap_delta(xcord, ycord, order, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        # TERM: no strictly improving neighbour => strict local optimum.
        if best_i < 0:
            break

        # ACCEPT: only a strictly decreasing move is applied.
        accepted += 1
        if log is not None:
            log.append((accepted, current_cost, best_i, best_j))
        order[best_i], order[best_j] = order[best_j], order[best_i]
        current_cost += best_delta

        # COST invariant: incremental value must match the exact tour cost.
        exact_cost = tour_cost(xcord, ycord, order)
        assert math.isclose(current_cost, exact_cost, rel_tol=1e-9,
                            abs_tol=1e-9), (current_cost, exact_cost)
        # Non-negative distance sanity invariant.
        assert current_cost >= -1e-9, current_cost

    return initial_cost, current_cost, accepted


def solve(xcord, ycord, max_iterations=200, seed=0, restarts=0):
    """Run the hill climb with optional seeded random restarts.

    Always includes the identity-order start (the historical behaviour),
    followed by ``restarts`` starts generated from a *local*
    ``random.Random(seed)`` instance -- never the global RNG -- so equal
    seeds produce bitwise equal trajectories and different seeds do not
    interact with other consumers of the random module.

    Returns the best run as
    ``(start_order, final_order, initial_cost, final_cost, log)``.
    """
    n = len(xcord)
    rng = random.Random(seed)

    best = None
    starts = [list(range(n))]
    for _ in range(restarts):
        permutation = list(range(n))
        rng.shuffle(permutation)
        starts.append(permutation)

    for start in starts:
        order = list(start)
        run_log = []
        initial_cost, final_cost, _ = hill_climb(
            xcord, ycord, order, max_iterations, run_log)
        if best is None or final_cost < best[3]:
            best = (start, order, initial_cost, final_cost, run_log)
    return best


def parse_instance(text):
    """Parse instance text into ``(n, xcord, ycord)``.

    Raises ``ValueError`` with an explicit message on any malformed input
    instead of silently truncating or producing a bogus answer:
    empty input, non-positive n, missing coordinate line, coordinate
    count different from n, non-numeric token, or NaN/infinite value.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty input: expected number of cities on the "
                         "first line")

    first = lines[0].split()
    if len(first) != 1:
        raise ValueError("first line must contain a single integer n, "
                         "got %r" % lines[0])
    try:
        n = int(first[0])
    except ValueError:
        raise ValueError("invalid number of cities: %r" % lines[0])
    if n < 1:
        raise ValueError("number of cities must be positive, got %d" % n)

    if len(lines) < 3:
        raise ValueError("expected 3 lines (n, x coordinates, "
                         "y coordinates), got %d" % len(lines))

    def parse_coordinates(line, name):
        tokens = line.split()
        if len(tokens) != n:
            raise ValueError("expected %d %s coordinates, got %d"
                             % (n, name, len(tokens)))
        values = []
        for token in tokens:
            try:
                value = float(token)
            except ValueError:
                raise ValueError("non-numeric %s coordinate: %r"
                                 % (name, token))
            if not math.isfinite(value):
                raise ValueError("%s coordinates must be finite, got %r"
                                 % (name, token))
            values.append(value)
        return values

    xcord = parse_coordinates(lines[1], "x")
    ycord = parse_coordinates(lines[2], "y")
    return n, xcord, ycord


def render_run(start_order, order, initial_cost, final_cost, run_log):
    """Render one run using the historical console fields."""
    lines = []
    for iteration, cost, swap_i, swap_j in run_log:
        lines.append("iteration =  %d" % iteration)
        lines.append("total distance = %.2f" % cost)
        lines.append("swapped  %d %d" % (swap_i, swap_j))
    lines.append("finish")
    lines.append("Final order =  %s" % order)
    lines.append("Initial cost =  %s" % initial_cost)
    lines.append("Final cost =  %s" % final_cost)
    return "\n".join(lines) + "\n"


def plot_run(xcord, ycord, start_order, order, run_log, final_cost):
    """Optional animated replay; matplotlib is imported only on request."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    current = list(start_order)
    _, ax = plt.subplots()
    line, = ax.plot(
        [xcord[c] for c in current] + [xcord[current[0]]],
        [ycord[c] for c in current] + [ycord[current[0]]],
        "ro-")

    def draw(finished, iteration, cost):
        xs = [xcord[c] for c in current] + [xcord[current[0]]]
        ys = [ycord[c] for c in current] + [ycord[current[0]]]
        line.set_xdata(xs)
        line.set_ydata(ys)
        title = "Final " if finished else "iteration no. %d \n" % iteration
        title = title + ("total distance = %.2f \n" % cost)
        ax.set_title(title, fontsize=15)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)

    for iteration, cost, swap_i, swap_j in run_log:
        draw(False, iteration, cost)
        current[swap_i], current[swap_j] = current[swap_j], current[swap_i]
    draw(True, 0, final_cost)
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hill-climbing TSP solver")
    parser.add_argument("input", nargs="?", default="input.txt",
                        help="instance file (default: input.txt)")
    parser.add_argument("--max-iterations", type=int, default=200,
                        help="accepted-swap budget per run (default: 200)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for random restarts (default: 0)")
    parser.add_argument("--restarts", type=int, default=0,
                        help="number of seeded random-restart starts "
                             "(default: 0)")
    parser.add_argument("--plot", action="store_true",
                        help="animate the search with matplotlib")
    args = parser.parse_args(argv)

    if args.max_iterations < 0:
        print("error: --max-iterations must be non-negative, got %d"
              % args.max_iterations, file=sys.stderr)
        return 2
    if args.restarts < 0:
        print("error: --restarts must be non-negative, got %d"
              % args.restarts, file=sys.stderr)
        return 2

    try:
        with open(args.input, "r") as handle:
            text = handle.read()
    except OSError as exc:
        print("error: cannot read %r: %s" % (args.input, exc),
              file=sys.stderr)
        return 2

    try:
        _, xcord, ycord = parse_instance(text)
    except ValueError as exc:
        print("error: invalid instance: %s" % exc, file=sys.stderr)
        return 2

    start_order, order, initial_cost, final_cost, run_log = solve(
        xcord, ycord,
        max_iterations=args.max_iterations,
        seed=args.seed,
        restarts=args.restarts)

    output = render_run(start_order, order, initial_cost,
                        final_cost, run_log)
    sys.stdout.write(output)

    if args.plot:
        plot_run(xcord, ycord, start_order, order, run_log, final_cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
