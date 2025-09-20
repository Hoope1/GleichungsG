import random
import sys
from fractions import Fraction
from pathlib import Path

import sympy as sp

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gleichungs_generator import (  # noqa: E402
    DEFAULT_CONFIG,
    Equation,
    build_L4,
    fraction_to_mixed,
    generate_equations,
    numeric_limits_ok,
    sample_solution,
    solve_steps,
    x,
)


def test_build_L4_mixed_format_and_steps():
    rng = random.Random(0)
    sol = sample_solution(rng, DEFAULT_CONFIG)
    eq = build_L4(rng, sol, DEFAULT_CONFIG)
    assert "*" not in eq.text
    steps = solve_steps(eq, DEFAULT_CONFIG)
    assert steps[0].description_de == "Ausgangsgleichung"


def test_generate_equations_all_levels():
    cfg = DEFAULT_CONFIG.copy()
    cfg["seed"] = 0
    cfg["counts"] = {"L1": 1, "L2": 1, "L3": 1, "L4": 1, "L5": 1}
    problems = generate_equations(cfg)
    assert len(problems) == 5
    assert {p.equation.level for p in problems} == {"L1", "L2", "L3", "L4", "L5"}


def test_fraction_to_mixed_handles_negative_values():
    assert fraction_to_mixed(Fraction(-5, 3)) == "-1 2/3"
    assert fraction_to_mixed(Fraction(-4, 2)) == "-2"


def test_solve_steps_handles_symbolic_denominator():
    eq = Equation(
        level="L5",
        sympy_eq=sp.Eq((2 * x + 3) / (3 * x + 5), sp.Rational(7, 11)),
        text="",
        solution=Fraction(2, 1),
        excluded={Fraction(-5, 3)},
    )

    assert numeric_limits_ok(eq)
    steps = solve_steps(eq, DEFAULT_CONFIG)
    assert any("Hauptnenner" in step.description_de for step in steps)
    assert steps[-1].rhs == sp.Integer(2)
