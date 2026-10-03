"""Regression and invariant tests for newmodified.py (stdlib unittest).

Run with:  python3 -m unittest test_newmodified -v
"""

import math
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from itertools import permutations

import newmodified as tsp

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "newmodified.py")


def make_instance(n, seed, lo=0, hi=1000):
    rng = random.Random(seed)
    xs = [rng.randrange(lo, hi) for _ in range(n)]
    ys = [rng.randrange(lo, hi) for _ in range(n)]
    return xs, ys


def instance_text(xs, ys):
    n = len(xs)
    return "%d\n%s\n%s\n" % (
        n, " ".join(map(str, xs)), " ".join(map(str, ys)))


def brute_force_optimal(xs, ys):
    """Exact optimum over all tours, fixing city 0 to kill rotation."""
    n = len(xs)
    best = math.inf
    for perm in permutations(range(1, n)):
        cost = tsp.tour_cost(xs, ys, [0] + list(perm))
        if cost < best:
            best = cost
    return best


def run_cli(text, *args):
    """Run the CLI on a temp instance file; return CompletedProcess."""
    with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False) as handle:
        handle.write(text)
        path = handle.name
    try:
        return subprocess.run(
            [sys.executable, SCRIPT, path, *args],
            capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(path)


def parse_output(stdout):
    """Parse the historical console fields into a structured record."""
    iterations = re.findall(r"^iteration =  (\d+)$", stdout, re.M)
    distances = re.findall(r"^total distance = (\S+)$", stdout, re.M)
    swaps = re.findall(r"^swapped  (\d+) (\d+)$", stdout, re.M)
    finishes = re.findall(r"^finish$", stdout, re.M)
    order = re.search(r"^Final order =  \[(.*)\]$", stdout, re.M)
    initial = re.search(r"^Initial cost =  (\S+)$", stdout, re.M)
    final = re.search(r"^Final cost =  (\S+)$", stdout, re.M)
    return {
        "iterations": [int(v) for v in iterations],
        "distances": [float(v) for v in distances],
        "swaps": [(int(a), int(b)) for a, b in swaps],
        "finishes": finishes,
        "order": ([int(v) for v in order.group(1).split(", ")]
                  if order else None),
        "initial": float(initial.group(1)) if initial else None,
        "final": float(final.group(1)) if final else None,
    }


class SwapDeltaExactnessTest(unittest.TestCase):
    """The acceptance delta must equal the true cost difference for all n."""

    def test_delta_matches_recomputation_for_all_pairs(self):
        for n in range(2, 10):
            xs, ys = make_instance(n, seed=50 + n)
            for trial in range(5):
                rng = random.Random(9000 + 100 * n + trial)
                order = list(range(n))
                rng.shuffle(order)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        before = tsp.tour_cost(xs, ys, order)
                        swapped = list(order)
                        swapped[i], swapped[j] = swapped[j], swapped[i]
                        after = tsp.tour_cost(xs, ys, swapped)
                        delta = tsp.swap_delta(xs, ys, order, i, j)
                        self.assertTrue(
                            math.isclose(delta, after - before,
                                         rel_tol=1e-9, abs_tol=1e-9),
                            "n=%d pair=(%d,%d) delta=%r exact=%r"
                            % (n, i, j, delta, after - before))

    def test_two_city_delta_is_exactly_zero(self):
        # Regression: the old heuristic double-counted the single edge of
        # a 2-city tour and drove the running cost negative.
        xs, ys = [0.0, 3.0], [0.0, 4.0]
        self.assertEqual(tsp.swap_delta(xs, ys, [0, 1], 0, 1), 0.0)


class InvariantTest(unittest.TestCase):
    """Acceptance and termination invariants, checked on solver output."""

    def check_invariants(self, xs, ys, seed=0, restarts=0):
        start, order, initial, final, log = tsp.solve(
            xs, ys, max_iterations=200, seed=seed, restarts=restarts)
        n = len(xs)

        # COST: reported costs equal the exact tour costs.
        self.assertEqual(initial, tsp.tour_cost(xs, ys, start))
        self.assertTrue(math.isclose(
            final, tsp.tour_cost(xs, ys, order),
            rel_tol=1e-9, abs_tol=1e-9))
        self.assertEqual(len(order), n)
        self.assertEqual(sorted(order), list(range(n)))
        if log:
            self.assertEqual(log[0][1], initial)

        # ACCEPT: strictly decreasing cost chain across accepted swaps.
        chain = [entry[1] for entry in log] + [final]
        for prev, nxt in zip(chain, chain[1:]):
            self.assertLess(nxt, prev)
        self.assertLessEqual(final, initial)

        # Non-negative distances => non-negative cost (no negative drift).
        self.assertGreaterEqual(final, -1e-9)

        # TERM: on early exit the final order is a strict local optimum.
        if len(log) < 200:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    self.assertGreaterEqual(
                        tsp.swap_delta(xs, ys, order, i, j), -1e-9)
        return order, final

    def test_invariants_hold_across_instances(self):
        for n in (1, 2, 3, 5, 8, 12):
            xs, ys = make_instance(n, seed=300 + n)
            self.check_invariants(xs, ys)
            self.check_invariants(xs, ys, seed=11, restarts=5)

    def test_invariants_from_cli_output(self):
        xs, ys = make_instance(10, seed=42)
        proc = run_cli(instance_text(xs, ys))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = parse_output(proc.stdout)

        # Historical fields are all present.
        self.assertIsNotNone(out["order"])
        self.assertIsNotNone(out["initial"])
        self.assertIsNotNone(out["final"])

        # Strictly decreasing acceptance chain, final cost consistent.
        chain = out["distances"] + [out["final"]]
        for prev, nxt in zip(chain, chain[1:]):
            self.assertLess(nxt, prev)
        self.assertTrue(math.isclose(
            out["final"], tsp.tour_cost(xs, ys, out["order"]),
            rel_tol=1e-9, abs_tol=1e-9))


class DeterminismTest(unittest.TestCase):
    """A fixed seed must give bitwise-identical results."""

    def test_same_seed_bitwise_identical_cli(self):
        xs, ys = make_instance(15, seed=123)
        text = instance_text(xs, ys)
        args = ("--seed", "7", "--restarts", "10")
        first = run_cli(text, *args)
        second = run_cli(text, *args)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stderr, second.stderr)

    def test_same_seed_bitwise_identical_in_process(self):
        xs, ys = make_instance(12, seed=555)
        run_a = tsp.solve(xs, ys, seed=99, restarts=8)
        run_b = tsp.solve(xs, ys, seed=99, restarts=8)
        self.assertEqual(run_a, run_b)

    def test_seed_only_affects_restarts(self):
        xs, ys = make_instance(10, seed=77)
        plain_a = tsp.solve(xs, ys, seed=1, restarts=0)
        plain_b = tsp.solve(xs, ys, seed=2, restarts=0)
        self.assertEqual(plain_a, plain_b)


class OptimalityTest(unittest.TestCase):
    """Small instances are compared against the exhaustive optimum."""

    def test_small_instances_match_brute_force(self):
        for n in range(2, 9):
            xs, ys = make_instance(n, seed=1000 + n)
            optimal = brute_force_optimal(xs, ys)
            _, _, _, final, _ = tsp.solve(
                xs, ys, max_iterations=200, seed=7, restarts=25)
            # Validity: a heuristic tour can never beat the optimum.
            self.assertGreaterEqual(final, optimal - 1e-9)
            # On these fixed seeded instances the optimum is reached.
            self.assertTrue(
                math.isclose(final, optimal, rel_tol=1e-9, abs_tol=1e-9),
                "n=%d final=%r optimal=%r" % (n, final, optimal))

    def test_never_beats_optimum_without_restarts(self):
        for n in range(2, 9):
            xs, ys = make_instance(n, seed=2000 + n)
            optimal = brute_force_optimal(xs, ys)
            _, _, _, final, _ = tsp.solve(xs, ys, restarts=0)
            self.assertGreaterEqual(final, optimal - 1e-9)


class EdgeCaseTest(unittest.TestCase):
    def test_two_cities(self):
        xs, ys = [0.0, 3.0], [0.0, 4.0]
        _, order, initial, final, log = tsp.solve(xs, ys)
        # Tour goes out and back: 2 * 5.0; old code reported 0 then
        # negative costs and kept "improving" for 200 iterations.
        self.assertEqual(initial, 10.0)
        self.assertEqual(final, 10.0)
        self.assertEqual(log, [])
        self.assertEqual(final, brute_force_optimal(xs, ys))

    def test_two_cities_cli(self):
        proc = run_cli("2\n0 3\n0 4\n")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = parse_output(proc.stdout)
        self.assertEqual(out["final"], 10.0)
        self.assertEqual(out["iterations"], [])

    def test_single_city(self):
        _, order, initial, final, log = tsp.solve([5.0], [6.0])
        self.assertEqual((initial, final, log), (0.0, 0.0, []))
        self.assertEqual(order, [0])

    def test_identical_coordinates(self):
        xs = [7.0] * 6
        ys = [7.0] * 6
        _, _, initial, final, log = tsp.solve(xs, ys)
        self.assertEqual(initial, 0.0)
        self.assertEqual(final, 0.0)
        self.assertEqual(log, [])

    def test_duplicate_coordinates_mixed(self):
        xs = [0.0, 0.0, 4.0, 4.0, 2.0]
        ys = [0.0, 0.0, 0.0, 0.0, 3.0]
        _, order, _, final, _ = tsp.solve(xs, ys)
        self.assertGreaterEqual(final, 0.0)
        self.assertTrue(math.isclose(
            final, tsp.tour_cost(xs, ys, order),
            rel_tol=1e-9, abs_tol=1e-9))

    def test_no_extra_iteration_on_stop(self):
        # Exactly one "finish", and every printed iteration carries a swap:
        # the final non-improving scan must not appear as an extra round.
        xs, ys = make_instance(10, seed=42)
        proc = run_cli(instance_text(xs, ys))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = parse_output(proc.stdout)
        self.assertEqual(len(out["finishes"]), 1)
        self.assertEqual(len(out["iterations"]), len(out["swaps"]))
        self.assertEqual(out["iterations"],
                         list(range(1, len(out["iterations"]) + 1)))
        self.assertTrue(proc.stdout.rstrip().endswith(
            "Final cost =  %s" % out["final"]))


class InvalidInputTest(unittest.TestCase):
    """Abnormal input must fail loudly, never silently return a result."""

    def check_cli_error(self, text, needle):
        proc = run_cli(text)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("error", proc.stderr)
        self.assertIn(needle, proc.stderr)
        self.assertEqual(proc.stdout, "")

    def test_empty_file(self):
        self.check_cli_error("", "empty input")

    def test_zero_cities(self):
        self.check_cli_error("0\n\n\n", "must be positive")

    def test_negative_city_count(self):
        self.check_cli_error("-3\n1 2 3\n1 2 3\n", "must be positive")

    def test_non_integer_n(self):
        self.check_cli_error("ten\n1 2\n1 2\n", "invalid number of cities")

    def test_missing_coordinate_line(self):
        self.check_cli_error("2\n0 1\n", "expected 3 lines")

    def test_coordinate_count_mismatch(self):
        self.check_cli_error("3\n0 1\n0 1 2\n",
                             "expected 3 x coordinates, got 2")

    def test_extra_coordinates_rejected(self):
        self.check_cli_error("2\n0 1 2\n0 1\n",
                             "expected 2 x coordinates, got 3")

    def test_non_numeric_coordinate(self):
        self.check_cli_error("2\n0 abc\n0 1\n", "non-numeric x coordinate")

    def test_nan_coordinate(self):
        self.check_cli_error("2\n0 nan\n0 1\n", "must be finite")

    def test_infinite_coordinate(self):
        self.check_cli_error("2\n0 1\n0 inf\n", "must be finite")

    def test_missing_file(self):
        proc = subprocess.run(
            [sys.executable, SCRIPT, "/nonexistent/instance.txt"],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("cannot read", proc.stderr)

    def test_parse_instance_raises_value_error(self):
        with self.assertRaises(ValueError):
            tsp.parse_instance("0\n\n\n")
        with self.assertRaises(ValueError):
            tsp.parse_instance("2\n0 1\n0\n")


class CliCompatibilityTest(unittest.TestCase):
    def test_output_fields_unchanged(self):
        xs, ys = make_instance(10, seed=42)
        proc = run_cli(instance_text(xs, ys))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for field in ("iteration =  ", "total distance = ", "swapped  ",
                      "finish", "Final order =  ", "Initial cost =  ",
                      "Final cost =  "):
            self.assertIn(field, proc.stdout)

    def test_default_input_file(self):
        proc = subprocess.run(
            [sys.executable, SCRIPT], cwd=HERE,
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Final cost =  ", proc.stdout)


if __name__ == "__main__":
    unittest.main()
