import random
from fractions import Fraction
import sympy as sp
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from gleichungs_generator import build_L1, DEFAULT_CONFIG, x, sample_solution


def test_build_L1_solution():
    rng = random.Random(0)
    sol = sample_solution(rng, DEFAULT_CONFIG)
    eq = build_L1(rng, sol, DEFAULT_CONFIG)
    result = sp.solve(eq.sympy_eq, x)
    assert len(result) == 1
