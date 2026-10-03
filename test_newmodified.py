import itertools
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import newmodified as tsp


def brute_force_optimum(coords):
    """Exact optimum tour length by enumeration (fixing city 0 as anchor)."""
    n = len(coords)
    best = math.inf
    for perm in itertools.permutations(range(1, n)):
        order = (0,) + perm
        best = min(best, tsp.tour_length(coords, list(order)))
    return best


def write_instance(coords):
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as handle:
        handle.write("%d\n" % len(coords))
        handle.write(" ".join(repr(c[0]) for c in coords) + "\n")
        handle.write(" ".join(repr(c[1]) for c in coords) + "\n")
    return path


def write_raw(text):
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    return path


def run_cli(*args):
    return subprocess.run(
        [sys.executable, os.path.join(HERE, "newmodified.py"), *args],
        capture_output=True, text=False, cwd=HERE,
    )


class EdgeCaseTests(unittest.TestCase):
    def test_two_cities_no_negative_cost(self):
        coords = [(0.0, 0.0), (3.0, 4.0)]
        result = tsp.hill_climb(coords, emit=lambda *a: None)
        self.assertAlmostEqual(result["final_cost"], 10.0)
        self.assertGreaterEqual(result["final_cost"], 0.0)
        self.assertEqual(result["iterations"], 0)
        self.assertTrue(result["finished"])

    def test_single_city(self):
        result = tsp.hill_climb([(5.0, -7.0)], emit=lambda *a: None)
        self.assertEqual(result["final_cost"], 0.0)
        self.assertEqual(result["order"], [0])

    def test_empty_city_list_rejected(self):
        with self.assertRaises(ValueError):
            tsp.hill_climb([], emit=lambda *a: None)
        path = write_raw("0\n\n\n")
        try:
            with self.assertRaisesRegex(ValueError, "at least 1"):
                tsp.read_instance(path)
        finally:
            os.unlink(path)

    def test_identical_coordinates_terminate_immediately(self):
        coords = [(2.0, 2.0)] * 6
        lines = []
        result = tsp.hill_climb(coords, emit=lines.append)
        self.assertEqual(result["final_cost"], 0.0)
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(lines, ["finish"])

    def test_negative_coordinates_supported(self):
        coords = [(-3.0, -4.0), (3.0, -4.0), (0.0, 5.0), (-1.0, 2.0)]
        result = tsp.hill_climb(coords, emit=lambda *a: None)
        self.assertGreaterEqual(result["final_cost"], 0.0)
        self.assertAlmostEqual(result["final_cost"], brute_force_optimum(coords))

    def test_no_extra_round_at_stop(self):
        # Every printed iteration must correspond to an accepted swap:
        # the stopping check must not run an additional empty round.
        coords = [(0.0, 0.0), (1.0, 5.0), (6.0, 2.0), (3.0, -4.0), (8.0, 8.0)]
        lines = []
        tsp.hill_climb(coords, emit=lines.append)
        iterations = sum(1 for l in lines if l.startswith("iteration ="))
        swaps = sum(1 for l in lines if l.startswith("swapped"))
        self.assertEqual(iterations, swaps)
        self.assertEqual(lines.count("finish"), 1)
        self.assertEqual(lines[-1], "finish")


class InvariantTests(unittest.TestCase):
    def test_acceptance_and_termination_invariant(self):
        rng = random.Random(1234)
        for _ in range(5):
            coords = [(rng.uniform(-100, 100), rng.uniform(-100, 100))
                      for _ in range(12)]
            costs = []
            def emit(line):
                if line.startswith("total distance"):
                    costs.append(float(line.split("=")[1]))
            result = tsp.hill_climb(coords, emit=emit)
            # (A1) strictly decreasing accepted costs.
            for before, after in zip(costs, costs[1:]):
                self.assertGreater(before, after)
            # (A2) tracked cost equals exact tour length and is non-negative.
            self.assertAlmostEqual(
                result["final_cost"], tsp.tour_length(coords, result["order"]))
            self.assertGreaterEqual(result["final_cost"], 0.0)
            # (T2) certified 2-swap local optimum.
            order = result["order"]
            n = len(order)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    delta = tsp.swap_delta(coords, order, i, j)
                    self.assertGreaterEqual(
                        delta, -tsp.TOL * max(1.0, result["final_cost"]))


class SeedStabilityTests(unittest.TestCase):
    def test_same_seed_bit_identical_output(self):
        instance = os.path.join(HERE, "input3.txt")
        outputs = []
        for _ in range(2):
            proc = run_cli(instance, "--seed", "42", "--shuffle", "--no-plot")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.append(proc.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_default_run_deterministic(self):
        instance = os.path.join(HERE, "input.txt")
        outputs = []
        for _ in range(2):
            proc = run_cli(instance, "--no-plot")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outputs.append(proc.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_shuffle_changes_start_but_stays_valid(self):
        coords = [(float(i * 7 % 13), float(i * 5 % 11)) for i in range(9)]
        result = tsp.solve(coords, seed=7, shuffle=True, plot=False,
                           emit=lambda *a: None)
        self.assertEqual(sorted(result["order"]), list(range(9)))


class BruteForceComparisonTests(unittest.TestCase):
    def test_small_instances_against_exhaustive_optimum(self):
        rng = random.Random(99)
        for n in range(1, 9):
            coords = [(rng.uniform(-50, 50), rng.uniform(-50, 50))
                      for _ in range(n)]
            result = tsp.hill_climb(coords, emit=lambda *a: None)
            optimum = brute_force_optimum(coords)
            # A heuristic may never beat the exact optimum.
            self.assertGreaterEqual(result["final_cost"], optimum - 1e-6)
            # For n <= 3 every tour is optimal, so equality must hold.
            if n <= 3:
                self.assertAlmostEqual(result["final_cost"], optimum)

    def test_square_instance_reaches_optimum(self):
        coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        result = tsp.hill_climb(coords, emit=lambda *a: None)
        self.assertAlmostEqual(result["final_cost"], 4.0)


class InvalidInputTests(unittest.TestCase):
    BAD_INPUTS = {
        "zero_cities": "0\n1 2\n3 4\n",
        "negative_cities": "-2\n1 2\n3 4\n",
        "non_integer_cities": "2.5\n1 2\n3 4\n",
        "missing_y_line": "2\n1 2\n",
        "too_few_x": "3\n1 2\n1 2 3\n",
        "too_many_y": "2\n1 2\n1 2 3\n",
        "non_numeric": "2\n1 abc\n3 4\n",
        "nan_coordinate": "2\n1 nan\n3 4\n",
        "empty_file": "",
    }

    def test_invalid_inputs_raise_clear_errors(self):
        for name, text in self.BAD_INPUTS.items():
            path = write_raw(text)
            try:
                with self.assertRaises(ValueError, msg=name):
                    tsp.read_instance(path)
            finally:
                os.unlink(path)

    def test_cli_reports_error_instead_of_silent_wrong_answer(self):
        for name, text in self.BAD_INPUTS.items():
            path = write_raw(text)
            try:
                proc = run_cli(path, "--no-plot")
                self.assertNotEqual(proc.returncode, 0, msg=name)
                self.assertIn(b"error", proc.stderr.lower(), msg=name)
                self.assertNotIn(b"Final cost", proc.stdout, msg=name)
            finally:
                os.unlink(path)

    def test_missing_file_reports_error(self):
        proc = run_cli("does_not_exist_12345.txt", "--no-plot")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"error", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
