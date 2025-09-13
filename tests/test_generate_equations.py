import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gleichungs_generator import (
    DEFAULT_CONFIG,
    build_L4,
    generate_equations,
    sample_solution,
    solve_steps,
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
