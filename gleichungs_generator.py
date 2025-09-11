#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Einfacher Gleichungs‑Generator für lineare Gleichungen mit einer Variablen.

Die Implementierung erzeugt zufällige Aufgaben der Level L1–L5 und
speichert sie als Arbeits- und Lösungsblatt im DOCX‑Format.  Sie folgt
den Vorgaben aus AGENTS.md in kompakter Form.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Set

import sympy as sp
from docx import Document
from docx.shared import Pt

x = sp.Symbol("x")

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    "seed": 0,
    "counts": {"L1": 4, "L2": 4, "L3": 6, "L4": 4, "L5": 5},
    "coeff_range": (-12, 12),
    "denom_range": (2, 12),
    "prefer_fraction": 0.85,
    "visual_complexity": "clean",
    "solutions": {"improper_and_mixed": True},
    "backend": "docx",
}

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def nice_frac(fr: Fraction) -> str:
    if fr.denominator == 1:
        return f"{fr.numerator}"
    return f"{fr.numerator}/{fr.denominator}"


def sample_solution(rng: random.Random, cfg: Dict) -> Fraction:
    a, b = cfg["coeff_range"]
    dmin, dmax = cfg["denom_range"]
    pref = cfg["prefer_fraction"]
    while True:
        p = rng.randint(a, b)
        q = rng.randint(dmin, dmax)
        if q == 0:
            continue
        r = Fraction(p, q)
        if rng.random() < pref and r.denominator == 1:
            continue
        return r


def expr_to_str(expr: sp.Expr) -> str:
    s = sp.sstr(expr)
    s = s.replace("*x", "x")
    s = s.replace("*", "")
    s = s.replace(" 1x", " x")
    s = s.replace("-1x", "-x")
    s = s.replace("+ -", "- ")
    return s


def poly_to_str(expr: sp.Expr) -> str:
    poly = sp.Poly(expr, x)
    A = Fraction(poly.coeffs()[0])
    B = Fraction(poly.coeffs()[1]) if len(poly.coeffs()) > 1 else Fraction(0)
    parts: List[str] = []
    if A == 1:
        parts.append("x")
    elif A == -1:
        parts.append("-x")
    else:
        parts.append(f"{nice_frac(A)}x")
    if B > 0:
        parts.append(f"+ {nice_frac(B)}")
    elif B < 0:
        parts.append(f"- {nice_frac(-B)}")
    return " ".join(parts) if parts else "0"


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class Equation:
    level: str
    sympy_eq: sp.Eq
    text: str
    solution: Fraction
    excluded: Set[Fraction] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Builder für Aufgabenlevel
# ---------------------------------------------------------------------------


def build_L1(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    mn, mx = cfg["coeff_range"]
    a = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    b = rng.randint(mn, mx)
    c = a * sol + b
    form = rng.choice([1, 2, 3])
    if form == 1:
        lhs = a * x + b
        rhs = sp.Integer(c)
        text = f"{a}x + {b} = {c}"
    elif form == 2:
        lhs = sp.Integer(c)
        rhs = a * x + b
        text = f"{c} = {a}x + {b}"
    else:
        lhs = b - a * x
        rhs = sp.Integer(c)
        text = f"{b} - {a}x = {c}"
    return Equation("L1", sp.Eq(lhs, rhs, evaluate=False), text, sol)


def build_L2(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    mn, mx = cfg["coeff_range"]
    while True:
        a = rng.choice([i for i in range(mn, mx + 1) if i != 0])
        d = rng.choice([i for i in range(mn, mx + 1) if i != a])
        e = rng.randint(mn, mx)
        b = d * sol + e - a * sol
        if b.denominator == 1:
            lhs = a * x + int(b)
            rhs = d * x + e
            text = f"{a}x + {int(b)} = {d}x + {e}"
            return Equation("L2", sp.Eq(lhs, rhs, evaluate=False), text, sol)


def build_L3(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    mn, mx = cfg["coeff_range"]
    pattern = rng.choice(["plus", "minus", "times"])
    m = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    n = rng.randint(mn, mx)
    p = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    q = rng.randint(mn, mx)
    if pattern == "plus":
        k = p * sol + q - (m * sol + n)
        lhs = k + (m * x + n)
        rhs = p * x + q
        text = f"{k} + ({m}x + {n}) = {p}x + {q}"
    elif pattern == "minus":
        k = p * sol + q + (m * sol + n)
        lhs = k - (m * x + n)
        rhs = p * x + q
        text = f"{k} - ({m}x + {n}) = {p}x + {q}"
    else:
        t = rng.choice([i for i in range(mn, mx + 1) if i != 0])
        q = t * (m * sol + n) - p * sol
        lhs = t * (m * x + n)
        rhs = p * x + q
        text = f"{t}({m}x + {n}) = {p}x + {q}"
    return Equation("L3", sp.Eq(lhs, rhs, evaluate=False), text, sol)


def build_L4(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    mn, mx = cfg["coeff_range"]
    a = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    m = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    n = rng.randint(mn, mx)
    b = rng.randint(mn, mx)
    c = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    d = rng.randint(mn, mx)
    p = rng.randint(mn, mx)
    q = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    r = rng.randint(mn, mx)
    s = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    t = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    u = (a * (m * sol + n) + b - (c * sol + d) - (p - (q * sol + r))) / s - t * sol
    lhs = a * (m * x + n) + b - (c * x + d)
    rhs = p - (q * x + r) + s * (t * x + u)
    text = f"{a}({m}x + {n}) + {b} - ({c}x + {d}) = {p} - ({q}x + {r}) + {s}({t}x + {u})"
    return Equation("L4", sp.Eq(lhs, rhs, evaluate=False), text, sol)



def build_L5(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    mn, mx = cfg["coeff_range"]
    while True:
        variant = rng.choice(["const", "two"])
        if variant == "const":
            a = rng.choice([i for i in range(mn, mx + 1) if i != 0])
            c = rng.choice([i for i in range(mn, mx + 1) if i != 0])
            d = rng.randint(mn, mx)
            k_num = rng.choice([i for i in range(mn, mx + 1) if i != 0])
            k_den = rng.choice([i for i in range(mn, mx + 1) if i != 0])
            k = Fraction(k_num, k_den)
            b = k * (c * sol + d) - a * sol
            if b.denominator != 1 or c * sol + d == 0:
                continue
            lhs = (a * x + int(b)) / (c * x + d)
            rhs = sp.Integer(k) if k.denominator == 1 else sp.Rational(k.numerator, k.denominator)
            text = f"({a}x + {int(b)})/({c}x + {d}) = {nice_frac(k)}"
            excluded = {Fraction(-d, c)}
            return Equation("L5", sp.Eq(lhs, rhs, evaluate=False), text, sol, excluded)
        else:
            a1 = rng.choice([i for i in range(mn, mx + 1) if i != 0])
            c1 = rng.choice([i for i in range(mn, mx + 1) if i != 0])
            d1 = rng.randint(mn, mx)
            a2 = rng.choice([i for i in range(mn, mx + 1) if i != 0])
            c2 = rng.choice([i for i in range(mn, mx + 1) if i != 0])
            d2 = rng.randint(mn, mx)
            b2 = rng.randint(mn, mx)
            num = (a2 * sol + b2) * (c1 * sol + d1)
            den = (c2 * sol + d2)
            if den == 0 or c1 * sol + d1 == 0:
                continue
            b1 = num / den - a1 * sol
            if isinstance(b1, Fraction) or getattr(b1, 'denominator', 1) != 1:
                b1 = Fraction(b1).limit_denominator()
            if b1.denominator != 1:
                continue
            lhs = (a1 * x + int(b1)) / (c1 * x + d1)
            rhs = (a2 * x + b2) / (c2 * x + d2)
            text = f"({a1}x + {int(b1)})/({c1}x + {d1}) = ({a2}x + {b2})/({c2}x + {d2})"
            excluded = {Fraction(-d1, c1), Fraction(-d2, c2)}
            return Equation("L5", sp.Eq(lhs, rhs, evaluate=False), text, sol, excluded)
BUILDERS = {
    "L1": build_L1,
    "L2": build_L2,
    "L3": build_L3,
    "L4": build_L4,
    "L5": build_L5,
}

# ---------------------------------------------------------------------------
# Validierung & Dedup
# ---------------------------------------------------------------------------



def validate(eq: Equation) -> bool:
    if not isinstance(eq.sympy_eq, sp.Equality):
        return False
    try:
        sol_list = sp.solve(eq.sympy_eq, x)
    except Exception:
        return False
    if len(sol_list) != 1:
        return False
    if sp.simplify(sol_list[0] - eq.solution) != 0:
        return False
    if any(sp.simplify(eq.sympy_eq.subs(x, ex)) == True for ex in eq.excluded):
        return False
    return True

def canonical_key(eq: Equation) -> str:
    expr = sp.together(eq.sympy_eq.lhs - eq.sympy_eq.rhs)
    expr = sp.expand(expr)
    expr = sp.collect(expr, x)
    return f"{sp.sstr(expr)}|{nice_frac(eq.solution)}"


# ---------------------------------------------------------------------------
# Solver mit Schrittverfolgung
# ---------------------------------------------------------------------------


def solve_steps(eq: Equation, cfg: Dict) -> List[str]:
    steps: List[str] = []
    if eq.excluded:
        ex = ", ".join(nice_frac(v) for v in sorted(eq.excluded))
        steps.append(f"Ausschlusswerte: x ≠ {ex}")
    steps.append(f"Ausgangsgleichung: {eq.text}")
    lhs = sp.expand(eq.sympy_eq.lhs)
    rhs = sp.expand(eq.sympy_eq.rhs)
    if lhs != eq.sympy_eq.lhs or rhs != eq.sympy_eq.rhs:
        steps.append(f"Klammern auflösen: {expr_to_str(lhs)} = {expr_to_str(rhs)}")
    lhs = sp.together(lhs)
    rhs = sp.together(rhs)
    lcm = sp.lcm(sp.denom(lhs), sp.denom(rhs))
    if lcm != 1:
        lhs = sp.simplify(lhs * lcm)
        rhs = sp.simplify(rhs * lcm)
        steps.append(f"Mit {expr_to_str(lcm)} multiplizieren: {expr_to_str(lhs)} = {expr_to_str(rhs)}")
    expr = sp.expand(lhs - rhs)
    steps.append(f"Alle Terme auf eine Seite: {expr_to_str(expr)} = 0")
    poly = sp.Poly(expr, x)
    A = Fraction(poly.all_coeffs()[0])
    B = Fraction(poly.all_coeffs()[1])
    steps.append(f"{nice_frac(A)}x = {nice_frac(-B)}")
    xval = -B / A
    steps.append(f"x = {nice_frac(xval)}")
    if cfg["solutions"]["improper_and_mixed"] and xval.denominator != 1:
        whole = xval.numerator // xval.denominator
        rest = abs(xval.numerator) % xval.denominator
        if whole != 0 and rest != 0:
            steps.append(f"x = {whole} {rest}/{xval.denominator}")
    return steps


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_arbeitsblatt(eqs: List[Equation], path: str):
    doc = Document()
    doc.add_heading("Lineare Gleichungen – Arbeitsblatt", level=1)
    idx = 1
    for level in ["L1", "L2", "L3", "L4", "L5"]:
        block = [e for e in eqs if e.level == level]
        if not block:
            continue
        doc.add_heading(f"Level {level[1]}", level=2)
        for e in block:
            p = doc.add_paragraph(f"{idx}. {e.text}")
            p.runs[0].font.size = Pt(12)
            idx += 1
            for _ in range(4):
                doc.add_paragraph(" ")
    doc.save(path)


def render_loesungen(eqs: List[Equation], path: str, cfg: Dict):
    doc = Document()
    doc.add_heading("Lineare Gleichungen – Lösungsblatt", level=1)
    idx = 1
    for level in ["L1", "L2", "L3", "L4", "L5"]:
        block = [e for e in eqs if e.level == level]
        if not block:
            continue
        doc.add_heading(f"Level {level[1]}", level=2)
        for e in block:
            doc.add_paragraph(f"Aufgabe {idx}: {e.text}")
            for s in solve_steps(e, cfg):
                p = doc.add_paragraph(f"• {s}")
                p.runs[0].font.size = Pt(11)
            idx += 1
            doc.add_paragraph(" ")
    doc.save(path)


# ---------------------------------------------------------------------------
# Haupt-Generator
# ---------------------------------------------------------------------------


def generate(cfg: Dict) -> List[Equation]:
    rng = random.Random(cfg["seed"])
    seen: Set[str] = set()
    results: List[Equation] = []
    for level, count in cfg["counts"].items():
        builder = BUILDERS[level]
        generated = 0
        while generated < count:
            sol = sample_solution(rng, cfg)
            eq = builder(rng, sol, cfg)
            if not validate(eq):
                continue
            key = canonical_key(eq)
            if key in seen:
                continue
            seen.add(key)
            results.append(eq)
            generated += 1
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator für lineare Gleichungen")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()
    cfg = DEFAULT_CONFIG.copy()
    cfg["seed"] = args.seed
    eqs = generate(cfg)
    render_arbeitsblatt(eqs, "arbeitsblatt.docx")
    render_loesungen(eqs, "loesungsblatt.docx", cfg)
    print("Dateien erstellt: arbeitsblatt.docx, loesungsblatt.docx")


if __name__ == "__main__":
    main()
