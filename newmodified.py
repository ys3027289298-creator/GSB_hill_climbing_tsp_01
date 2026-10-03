# Hill Climbing TSP solver (steepest-ascent 2-swap hill climbing).
import argparse
import math
import random
import sys

DEFAULT_MAX_ITERATIONS = 200

# Relative tolerance used for the acceptance invariant. A move is accepted
# only when it strictly improves the cost by more than TOL * scale, so plain
# floating-point reordering (e.g. identical coordinates) cannot be mistaken
# for progress.
TOL = 1e-9


def city_dist(coords, u, v):
    """Euclidean distance between cities u and v (always non-negative)."""
    xu, yu = coords[u]
    xv, yv = coords[v]
    return math.sqrt((xu - xv) ** 2 + (yu - yv) ** 2)


def tour_length(coords, order):
    """Exact length of the closed tour given by ``order``."""
    n = len(order)
    if n == 0:
        raise ValueError("cannot compute tour length for an empty city list")
    total = 0.0
    for i in range(n):
        total += city_dist(coords, order[i], order[(i + 1) % n])
    return total


def read_instance(path):
    """Read and validate an instance file.

    The file contains the number of cities on the first line followed by one
    line of x coordinates and one line of y coordinates. Any malformed input
    raises ValueError with an explicit message instead of silently producing
    wrong results.
    """
    with open(path, "r") as handle:
        lines = [line.strip() for line in handle.readlines()]
    lines = [line for line in lines if line != ""]
    if not lines:
        raise ValueError("empty input file: expected number of cities on the first line")

    header = lines[0].split()
    if len(header) != 1:
        raise ValueError("invalid number of cities: %r" % lines[0])
    try:
        n_float = float(header[0])
    except ValueError:
        raise ValueError("invalid number of cities: %r" % header[0])
    if not math.isfinite(n_float) or n_float != int(n_float):
        raise ValueError("number of cities must be a positive integer, got %r" % header[0])
    n = int(n_float)
    if n < 1:
        raise ValueError("number of cities must be at least 1, got %d" % n)

    if len(lines) < 2:
        raise ValueError("missing x coordinates line (expected %d numbers)" % n)
    if len(lines) < 3:
        raise ValueError("missing y coordinates line (expected %d numbers)" % n)

    def parse_coords(text, axis):
        tokens = text.split()
        if len(tokens) != n:
            raise ValueError(
                "expected %d %s coordinates, got %d" % (n, axis, len(tokens))
            )
        values = []
        for token in tokens:
            try:
                value = float(token)
            except ValueError:
                raise ValueError("invalid %s coordinate value: %r" % (axis, token))
            if not math.isfinite(value):
                raise ValueError("%s coordinate must be finite, got %r" % (axis, token))
            values.append(value)
        return values

    xcord = parse_coords(lines[1], "x")
    ycord = parse_coords(lines[2], "y")
    return list(zip(xcord, ycord))


def swap_delta(coords, order, i, j):
    """Cost change caused by swapping the cities at positions i, j (i < j).

    Only called when n >= 4. For n <= 3 every possible tour has identical
    length, so hill_climb skips neighbourhood search entirely (this is also
    the case that previously made a 2-city tour drift to a negative cost:
    its four affected tour edges collapse to only two).
    """
    n = len(order)
    if i == 0 and j == n - 1:
        # Normalize the pair adjacent across the tour wrap.
        i, j = j, i
    prev_i = (i - 1 + n) % n
    next_i = (i + 1) % n
    prev_j = (j - 1 + n) % n
    next_j = (j + 1) % n

    delta = 0.0
    if j != (i + 1) % n:
        # Non-adjacent positions: edges (i, i+1) and (j-1, j) change too.
        delta -= city_dist(coords, order[i], order[next_i])
        delta -= city_dist(coords, order[j], order[prev_j])
        delta += city_dist(coords, order[j], order[next_i])
        delta += city_dist(coords, order[i], order[prev_j])
    delta -= city_dist(coords, order[i], order[prev_i])
    delta -= city_dist(coords, order[j], order[next_j])
    delta += city_dist(coords, order[i], order[next_j])
    delta += city_dist(coords, order[j], order[prev_i])
    return delta


def hill_climb(coords, order=None, max_iterations=DEFAULT_MAX_ITERATIONS, emit=None):
    """Run steepest-ascent 2-swap hill climbing.

    Returns a dict with keys: order, initial_cost, final_cost, iterations,
    finished (True iff the loop stopped at a local optimum rather than at
    the iteration cap).

    Acceptance and termination invariant
    ------------------------------------
    Let c_k be the tour length at iteration k.

      (A1) A swap is accepted iff the smallest 2-swap delta satisfies
           delta < -TOL * max(1, c_k). Thus c_{k+1} < c_k by a fixed
           relative margin: the sequence of accepted costs is strictly
           decreasing (never equal, never larger).
      (A2) After every accepted swap the cost is recomputed exactly with
           tour_length(), so the tracked cost always equals the real tour
           length (no incremental floating-point drift) and stays >= 0.
      (T1) There are only n! distinct tours, and (A1) forbids revisiting a
           tour, so the number of accepted moves is finite; the loop ends in
           at most max_iterations iterations without an extra empty round.
      (T2) When the loop exits with finished=True, every 2-swap neighbour
           has delta >= -TOL * max(1, c): the returned tour is a certified
           local optimum of the 2-swap neighbourhood.
    """
    if emit is None:
        emit = lambda text="": print(text)

    n = len(coords)
    if n < 1:
        raise ValueError("number of cities must be at least 1, got 0")
    if order is None:
        order = list(range(n))
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of the %d cities" % n)
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    current_cost = tour_length(coords, order)
    initial_cost = current_cost
    finished = True
    iterations = 0

    for _ in range(max_iterations):
        # Evaluate the whole neighbourhood first; if no improving move
        # exists we stop immediately, without running an extra empty round.
        best_delta = 0.0
        best_i = best_j = -1
        if n > 3:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    delta = swap_delta(coords, order, i, j)
                    if delta < best_delta - TOL * max(1.0, abs(current_cost)):
                        best_delta = delta
                        best_i, best_j = i, j

        if best_i == -1:
            # (T2): no accepted move => local optimum.
            finished = True
            break

        iterations += 1
        emit("iteration =  %d" % iterations)
        emit("total distance = %.2f" % current_cost)

        order[best_i], order[best_j] = order[best_j], order[best_i]
        new_cost = tour_length(coords, order)  # (A2): exact recomputation

        scale = max(1.0, abs(current_cost))
        if not (new_cost < current_cost - TOL * scale):
            raise AssertionError(
                "acceptance invariant violated: %r not strictly below %r"
                % (new_cost, current_cost)
            )
        if not abs(new_cost - (current_cost + best_delta)) <= TOL * scale:
            raise AssertionError(
                "incremental delta disagrees with exact tour length"
            )
        if new_cost < -TOL:
            raise AssertionError("tour length became negative: %r" % new_cost)

        current_cost = new_cost
        finished = False
        emit("swapped  %d %d" % (best_i, best_j))
    else:
        finished = False

    if finished:
        emit("finish")

    return {
        "order": order,
        "initial_cost": initial_cost,
        "final_cost": current_cost,
        "iterations": iterations,
        "finished": finished,
    }


class Plotter:
    """Optional live visualisation; silently disabled without matplotlib."""

    def __init__(self, coords, order, enabled):
        self.enabled = enabled
        self.plt = None
        self.line = None
        self.coords = coords
        if not enabled:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            self.enabled = False
            return
        self.plt = plt
        xcord = [coords[k][0] for k in order]
        ycord = [coords[k][1] for k in order]
        self.line, = plt.gca().plot(xcord, ycord, "ro-")

    def _set_data(self, order):
        xcord = [self.coords[k][0] for k in order] + [self.coords[order[0]][0]]
        ycord = [self.coords[k][1] for k in order] + [self.coords[order[0]][1]]
        self.line.set_xdata(xcord)
        self.line.set_ydata(ycord)

    def draw(self, order, finished, iteration, cost):
        if not self.enabled:
            return
        self._set_data(order)
        title = "Final " if finished else "iteration no. %d \n" % iteration
        title = title + ("total distance = %.2f \n" % cost)
        self.plt.title(title, fontsize=15)
        self.plt.draw()
        self.plt.pause(0.1)

    def show(self):
        if self.enabled:
            self.plt.show()


def solve(coords, seed=0, shuffle=False, max_iterations=DEFAULT_MAX_ITERATIONS,
          plot=False, emit=None):
    """Entry point with a deterministic, seedable (but by default unused) RNG.

    The search itself is fully deterministic; ``shuffle`` uses an isolated
    ``random.Random(seed)`` so the same seed reproduces identical output bit
    for bit.
    """
    if emit is None:
        emit = lambda text="": print(text)

    order = list(range(len(coords)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(order)

    plotter = Plotter(coords, order, plot)
    result = hill_climb(coords, order, max_iterations=max_iterations, emit=emit)
    plotter.draw(result["order"], True, result["iterations"], result["final_cost"])

    emit("Final order =  %s" % result["order"])
    emit("Initial cost =  %s" % result["initial_cost"])
    emit("Final cost =  %s" % result["final_cost"])
    plotter.show()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hill Climbing TSP solver")
    parser.add_argument("input", nargs="?", default="input.txt",
                        help="instance file (default: input.txt)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed used with --shuffle (default: 0)")
    parser.add_argument("--shuffle", action="store_true",
                        help="start from a seeded random tour")
    parser.add_argument("--max-iterations", type=int,
                        default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--no-plot", action="store_true",
                        help="disable graphical output")
    args = parser.parse_args(argv)

    try:
        coords = read_instance(args.input)
        if args.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        solve(coords, seed=args.seed, shuffle=args.shuffle,
              max_iterations=args.max_iterations, plot=not args.no_plot)
    except (ValueError, OSError) as err:
        print("error: %s" % err, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
