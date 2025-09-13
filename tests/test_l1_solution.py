import random
import sys
from pathlib import Path

import sympy as sp

sys.path.append(str(Path(__file__).resolve().parents[1]))
from gleichungs_generator import DEFAULT_CONFIG, build_L1, sample_solution, x


def test_build_L1_solution():
    rng = random.Random(0)
    sol = sample_solution(rng, DEFAULT_CONFIG)
    eq = build_L1(rng, sol, DEFAULT_CONFIG)
    result = sp.solve(eq.sympy_eq, x)
    assert len(result) == 1
