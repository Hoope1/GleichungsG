#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gleichungs-Generator für lineare Gleichungen mit einer Variablen.

Generiert Aufgaben in 5 Schwierigkeitsleveln mit Lösungen als Brüche.
Erstellt Arbeits- und Lösungsblatt im DOCX-Format.
"""

from __future__ import annotations

import argparse
import math
import random
import re
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

MAX_ABS = 120
MAX_RESAMPLES = 200

DEFAULT_CONFIG: Dict = {
    "seed": 12345,
    "counts": {"L1": 4, "L2": 4, "L3": 6, "L4": 4, "L5": 5},
    "coeff_range": (-12, 12),
    "denom_range": (2, 12),
    "prefer_fraction": 0.85,
    "visual_complexity": "mixed",  # "clean" oder "mixed"
    "solutions": {"improper_and_mixed": True},
    "backend": "docx",
}

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def nice_frac(fr: Fraction) -> str:
    """Formatiert einen Bruch als String."""
    if fr.denominator == 1:
        return str(fr.numerator)
    return f"{fr.numerator}/{fr.denominator}"


def fraction_to_mixed(fr: Fraction) -> str:
    """Konvertiert Bruch in gemischte Zahl (für positive und negative)."""
    if fr.denominator == 1:
        return str(fr.numerator)

    whole = fr.numerator // fr.denominator
    remainder = abs(fr.numerator) - abs(whole) * fr.denominator

    if remainder == 0:
        return str(whole)

    if whole == 0:
        return nice_frac(fr)

    # Gemischte Zahl
    sign = "-" if fr < 0 else ""
    return f"{sign}{abs(whole)} {remainder}/{fr.denominator}"


def sample_solution(rng: random.Random, cfg: Dict) -> Fraction:
    """Zieht eine Ziellösung, bevorzugt Brüche."""
    a, b = cfg["coeff_range"]
    dmin, dmax = cfg["denom_range"]
    pref = cfg["prefer_fraction"]

    attempts = 0
    while attempts < 50:
        p = rng.randint(a, b)
        if p == 0:
            attempts += 1
            continue
        q = rng.randint(dmin, dmax)
        r = Fraction(p, q)

        # Bevorzuge nicht-ganzzahlige Lösungen
        if r.denominator == 1 and rng.random() < pref:
            attempts += 1
            continue

        # Prüfe ob Lösung im erlaubten Bereich
        if abs(r.numerator) <= MAX_ABS and abs(r.denominator) <= MAX_ABS:
            return r
        attempts += 1

    # Fallback: irgendeine gültige Lösung
    return Fraction(rng.randint(1, 12), rng.randint(2, 6))


def sympy_to_text(expr: sp.Expr, visual_complexity: str = "mixed") -> str:
    """Konvertiert SymPy-Ausdruck in lesbaren Text mit korrekter Formatierung."""
    s = str(expr)

    if visual_complexity == "mixed":
        # Implizites Mal (kein * Zeichen)
        s = s.replace("*x", "x")
        s = s.replace("*", "")
        s = s.replace(" 1*x", " x")
        s = s.replace("-1*x", "-x")
        s = s.replace("(1)*x", "x")
        s = s.replace("(-1)*x", "-x")
    else:  # clean
        s = s.replace("*", "·")

    # Aufräumen
    s = s.replace("+ -", "- ")
    s = s.replace("- -", "+ ")
    s = s.replace("  ", " ")

    # SymPy Rational Darstellung korrigieren
    s = re.sub(r"Rational\((-?\d+),\s*(\d+)\)", r"\1/\2", s)

    return s.strip()


def _fraction_ok(fr: Fraction) -> bool:
    """Prüft ob Bruch innerhalb der Zahlengrenzen liegt."""
    return abs(fr.numerator) <= MAX_ABS and abs(fr.denominator) <= MAX_ABS


def _expr_ok(expr: sp.Expr) -> bool:
    """Prüft ob alle Zahlen im Ausdruck innerhalb der Grenzen liegen."""
    for atom in expr.atoms(sp.Rational):
        fr = Fraction(int(atom.p), int(atom.q))
        if not _fraction_ok(fr):
            return False
    return True


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class Equation:
    """Repräsentiert eine Gleichung."""

    level: str
    sympy_eq: sp.Eq
    text: str
    solution: Fraction
    excluded: Set[Fraction] = field(default_factory=set)
    template: str = ""
    params: Dict = field(default_factory=dict)


@dataclass
class SolveStep:
    """Ein Schritt im Lösungsweg."""

    description_de: str
    lhs: sp.Expr
    rhs: sp.Expr


@dataclass
class Problem:
    """Aufgabe mit Lösungsschritten."""

    equation: Equation
    steps: List[SolveStep]
    resamples: int = 0
    dops: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Numerische Validierung
# ---------------------------------------------------------------------------


def numeric_limits_ok(eq: Equation) -> bool:
    """Prüft ob alle Zahlen in der Gleichung innerhalb der Limits sind."""
    # Prüfe Ausgangsgleichung
    if not _expr_ok(eq.sympy_eq.lhs) or not _expr_ok(eq.sympy_eq.rhs):
        return False

    # Prüfe nach Expansion
    lhs_exp = sp.expand(eq.sympy_eq.lhs)
    rhs_exp = sp.expand(eq.sympy_eq.rhs)
    if not _expr_ok(lhs_exp) or not _expr_ok(rhs_exp):
        return False

    # Prüfe LCM
    expr = eq.sympy_eq.lhs - eq.sympy_eq.rhs
    expr_together = sp.together(expr)

    # Extrahiere alle Nenner
    denominators = []
    for term in sp.Add.make_args(expr_together):
        _, denom = sp.fraction(term)
        if denom != 1:
            denominators.append(int(denom))

    if denominators:
        lcm_val = denominators[0]
        for d in denominators[1:]:
            lcm_val = math.lcm(lcm_val, d)
        if lcm_val > MAX_ABS:
            return False

    # Prüfe finale Koeffizienten nach Vereinfachung
    expr_simplified = sp.expand(sp.together(expr))
    if expr_simplified.has(x):
        poly = sp.Poly(expr_simplified, x)
        coeffs = poly.all_coeffs()
        for c in coeffs:
            if abs(float(c)) > MAX_ABS:
                return False

    return True


# ---------------------------------------------------------------------------
# Beautifier für Textdarstellung
# ---------------------------------------------------------------------------


def beautify_equation(
    lhs: sp.Expr, rhs: sp.Expr, visual_complexity: str = "mixed"
) -> str:
    """Formatiert eine Gleichung schön für die Ausgabe."""
    lhs_str = sympy_to_text(lhs, visual_complexity)
    rhs_str = sympy_to_text(rhs, visual_complexity)

    # Spezielle Formatierungen für "mixed" Komplexität
    if visual_complexity == "mixed":
        # Ersetze 1*(...) durch (...)
        lhs_str = re.sub(r"\b1\s*\(", "(", lhs_str)
        rhs_str = re.sub(r"\b1\s*\(", "(", rhs_str)

        # Ersetze -1*(...) durch -(...)
        lhs_str = re.sub(r"-1\s*\(", "-(", lhs_str)
        rhs_str = re.sub(r"-1\s*\(", "-(", rhs_str)

    return f"{lhs_str} = {rhs_str}"


# ---------------------------------------------------------------------------
# Builder für Level 1: x auf einer Seite
# ---------------------------------------------------------------------------


def build_L1(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    """Erstellt Level 1 Gleichung: x nur auf einer Seite."""
    mn, mx = cfg["coeff_range"]

    # Wähle Form
    form = rng.choice(["ax_plus_b", "a_minus_bx", "c_equals_dx_plus_e"])

    if form == "ax_plus_b":
        # ax + b = c
        a = rng.choice([i for i in range(mn, mx + 1) if i != 0])
        b = rng.randint(mn, mx)
        c = a * sol + b

        if not _fraction_ok(Fraction(c)):
            # Retry mit kleineren Werten
            a = rng.choice([i for i in range(-6, 7) if i != 0])
            b = rng.randint(-6, 6)
            c = a * sol + b

        lhs = a * x + b
        rhs = c
        text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))

    elif form == "a_minus_bx":
        # a - bx = c
        b = rng.choice([i for i in range(mn, mx + 1) if i != 0])
        a = rng.randint(mn, mx)
        c = a - b * sol

        if not _fraction_ok(Fraction(c)):
            b = rng.choice([i for i in range(-6, 7) if i != 0])
            a = rng.randint(-6, 6)
            c = a - b * sol

        lhs = a - b * x
        rhs = c
        text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))

    else:  # c_equals_dx_plus_e
        # c = dx + e
        d = rng.choice([i for i in range(mn, mx + 1) if i != 0])
        e = rng.randint(mn, mx)
        c = d * sol + e

        if not _fraction_ok(Fraction(c)):
            d = rng.choice([i for i in range(-6, 7) if i != 0])
            e = rng.randint(-6, 6)
            c = d * sol + e

        lhs = c
        rhs = d * x + e
        text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))

    return Equation("L1", sp.Eq(lhs, rhs, evaluate=False), text, sol)


# ---------------------------------------------------------------------------
# Builder für Level 2: x auf beiden Seiten
# ---------------------------------------------------------------------------


def build_L2(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    """Erstellt Level 2 Gleichung: x auf beiden Seiten."""
    mn, mx = cfg["coeff_range"]

    attempts = 0
    while attempts < 50:
        a = rng.choice([i for i in range(mn, mx + 1) if i != 0])
        d = rng.choice([i for i in range(mn, mx + 1) if i != 0 and i != a])
        e = rng.randint(mn, mx)
        b = d * sol + e - a * sol

        if _fraction_ok(Fraction(b)):
            lhs = a * x + b
            rhs = d * x + e
            text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
            return Equation("L2", sp.Eq(lhs, rhs, evaluate=False), text, sol)
        attempts += 1

    # Fallback
    lhs = 2 * x + 3
    rhs = x + 3 + sol
    text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
    return Equation("L2", sp.Eq(lhs, rhs, evaluate=False), text, sol)


# ---------------------------------------------------------------------------
# Builder für Level 3: Eine Klammer (je 2× +, -, ×)
# ---------------------------------------------------------------------------


def build_L3(
    rng: random.Random, sol: Fraction, cfg: Dict, pattern_index: int = None
) -> Equation:
    """Erstellt Level 3 Gleichung: Eine Klammer.

    pattern_index: 0,1 = plus; 2,3 = minus; 4,5 = times
    """
    mn, mx = -8, 8  # Kleinere Werte für Klammern

    # Bestimme Pattern basierend auf Index
    if pattern_index is None:
        pattern = rng.choice(["plus", "minus", "times"])
    else:
        if pattern_index < 2:
            pattern = "plus"
        elif pattern_index < 4:
            pattern = "minus"
        else:
            pattern = "times"

    m = rng.choice([i for i in range(mn, mx + 1) if i != 0])
    n = rng.randint(mn, mx)
    p = rng.choice([i for i in range(mn, mx + 1) if i != 0])

    if pattern == "plus":
        # k + (mx + n) = px + q
        k = rng.randint(mn, mx)
        q = k + (m * sol + n) - p * sol

        if _fraction_ok(Fraction(q)):
            lhs = k + (m * x + n)
            rhs = p * x + q
            text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
            template = "k + (mx + n) = px + q"
        else:
            # Fallback
            lhs = 2 + (x + 1)
            rhs = 2 * x + 1 + 2 - sol
            text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
            template = "plus"

    elif pattern == "minus":
        # k - (mx + n) = px + q
        k = rng.randint(mn, mx)
        q = k - (m * sol + n) - p * sol

        if _fraction_ok(Fraction(q)):
            lhs = k - (m * x + n)
            rhs = p * x + q
            text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
            template = "k - (mx + n) = px + q"
        else:
            # Fallback
            lhs = 5 - (x + 2)
            rhs = x + 3 - 2 * sol
            text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
            template = "minus"

    else:  # times
        # t(mx + n) = px + q
        t = rng.choice([i for i in range(-6, 7) if i != 0])
        q = t * (m * sol + n) - p * sol

        if _fraction_ok(Fraction(q)):
            lhs = t * (m * x + n)
            rhs = p * x + q

            # Bei "mixed" complexity: implizites Mal
            if cfg.get("visual_complexity", "mixed") == "mixed":
                lhs_str = f"{t}({sympy_to_text(m * x + n, 'mixed')})"
                rhs_str = sympy_to_text(rhs, "mixed")
                text = f"{lhs_str} = {rhs_str}"
            else:
                text = beautify_equation(
                    lhs, rhs, cfg.get("visual_complexity", "clean")
                )
            template = "t(mx + n) = px + q"
        else:
            # Fallback
            lhs = 3 * (x + 1)
            rhs = 4 * x + 3 - sol
            text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
            template = "times"

    return Equation("L3", sp.Eq(lhs, rhs, evaluate=False), text, sol, template=template)


# ---------------------------------------------------------------------------
# Builder für Level 4: Gemischte Klammern
# ---------------------------------------------------------------------------


def build_L4(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    """Erstellt Level 4 Gleichung: Gemischte Klammern."""
    # Kleinere Bereiche für komplexere Gleichungen
    attempts = 0
    while attempts < 50:
        a = rng.choice([i for i in range(-4, 5) if i != 0])
        m = rng.choice([i for i in range(-4, 5) if i != 0])
        n = rng.randint(-6, 6)
        b = rng.randint(-6, 6)
        c = rng.choice([i for i in range(-4, 5) if i != 0])
        d = rng.randint(-6, 6)
        p = rng.randint(-6, 6)
        q = rng.choice([i for i in range(-4, 5) if i != 0])
        r = rng.randint(-6, 6)
        s = rng.choice([i for i in range(-3, 4) if i != 0])
        t = rng.choice([i for i in range(-3, 4) if i != 0])

        # Berechne u so dass die Gleichung bei x=sol erfüllt ist
        left_at_sol = a * (m * sol + n) + b - (c * sol + d)
        right_without_u = p - (q * sol + r) + s * t * sol

        if s != 0:
            u = (left_at_sol - right_without_u) / s

            if _fraction_ok(Fraction(u)):
                lhs = a * (m * x + n) + b - (c * x + d)
                rhs = p - (q * x + r) + s * (t * x + u)

                # Formatierung für mixed complexity
                if cfg.get("visual_complexity", "mixed") == "mixed":
                    # Implizites Mal vor Klammern
                    lhs_str = f"{a}({sympy_to_text(m * x + n, 'mixed')}) + {b} - ({sympy_to_text(c * x + d, 'mixed')})"
                    rhs_str = f"{p} - ({sympy_to_text(q * x + r, 'mixed')}) + {s}({sympy_to_text(t * x + u, 'mixed')})"
                    text = f"{lhs_str} = {rhs_str}".replace("+ -", "- ")
                else:
                    text = beautify_equation(
                        lhs, rhs, cfg.get("visual_complexity", "clean")
                    )

                eq = Equation("L4", sp.Eq(lhs, rhs, evaluate=False), text, sol)
                if numeric_limits_ok(eq):
                    return eq
        attempts += 1

    # Fallback
    lhs = 2 * (x + 1) - (x - 1)
    rhs = x + 3
    text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
    return Equation("L4", sp.Eq(lhs, rhs, evaluate=False), text, sol)


# ---------------------------------------------------------------------------
# Builder für Level 5: Bruchgleichungen
# ---------------------------------------------------------------------------


def build_L5(rng: random.Random, sol: Fraction, cfg: Dict) -> Equation:
    """Erstellt Level 5 Gleichung: Bruchgleichungen."""
    variant = rng.choice(["simple", "two_fracs", "cross"])

    if variant == "simple":
        # (a/b)x = c
        a = rng.randint(1, 8)
        b = rng.randint(2, 8)
        c = Fraction(a, b) * sol

        if _fraction_ok(c):
            lhs = sp.Rational(a, b) * x
            rhs = (
                sp.Rational(c.numerator, c.denominator)
                if c.denominator != 1
                else c.numerator
            )
            text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
            return Equation("L5", sp.Eq(lhs, rhs, evaluate=False), text, sol)

    elif variant == "two_fracs":
        # (ax + b)/c + (dx + e)/f = g
        attempts = 0
        while attempts < 30:
            a = rng.choice([i for i in range(-4, 5) if i != 0])
            b = rng.randint(-6, 6)
            c = rng.randint(2, 8)
            d = rng.choice([i for i in range(-4, 5) if i != 0])
            e = rng.randint(-6, 6)
            f = rng.randint(2, 8)

            # Berechne g so dass die Gleichung bei x=sol erfüllt ist
            g = Fraction(a * sol + b, c) + Fraction(d * sol + e, f)

            if _fraction_ok(g) and math.lcm(c, f) <= MAX_ABS:
                lhs = (a * x + b) / c + (d * x + e) / f
                rhs = (
                    sp.Rational(g.numerator, g.denominator)
                    if g.denominator != 1
                    else g.numerator
                )
                text = beautify_equation(
                    lhs, rhs, cfg.get("visual_complexity", "mixed")
                )
                excluded = set()
                return Equation(
                    "L5", sp.Eq(lhs, rhs, evaluate=False), text, sol, excluded
                )
            attempts += 1

    else:  # cross
        # (ax + b)/(cx + d) = e/f
        attempts = 0
        while attempts < 30:
            a = rng.choice([i for i in range(-4, 5) if i != 0])
            b = rng.randint(-6, 6)
            c = rng.choice([i for i in range(-3, 4) if i != 0])
            d = rng.randint(-6, 6)

            # Prüfe ob sol ein Ausschlusswert wäre
            if c * sol + d == 0:
                attempts += 1
                continue

            # Berechne e/f
            val = Fraction(a * sol + b, c * sol + d)

            if _fraction_ok(val):
                e = val.numerator
                f = val.denominator

                lhs = (a * x + b) / (c * x + d)
                rhs = sp.Rational(e, f) if f != 1 else e
                text = beautify_equation(
                    lhs, rhs, cfg.get("visual_complexity", "mixed")
                )

                # Ausschlusswert
                excluded = {Fraction(-d, c)} if c != 0 else set()
                return Equation(
                    "L5", sp.Eq(lhs, rhs, evaluate=False), text, sol, excluded
                )
            attempts += 1

    # Fallback: einfache Bruchgleichung
    lhs = sp.Rational(1, 2) * x
    rhs = (
        sp.Rational(sol.numerator, 2 * sol.denominator)
        if sol.denominator != 1
        else sp.Rational(sol.numerator, 2)
    )
    text = beautify_equation(lhs, rhs, cfg.get("visual_complexity", "mixed"))
    return Equation("L5", sp.Eq(lhs, rhs, evaluate=False), text, sol)


# ---------------------------------------------------------------------------
# Validierung
# ---------------------------------------------------------------------------


def validate(eq: Equation) -> bool:
    """Validiert eine Gleichung."""
    try:
        # Löse die Gleichung
        solutions = sp.solve(eq.sympy_eq, x)

        # Muss genau eine Lösung haben
        if len(solutions) != 1:
            return False

        # Lösung muss mit Ziellösung übereinstimmen
        sol_sympy = solutions[0]
        sol_frac = Fraction(float(sol_sympy))

        if abs(sol_frac - eq.solution) > Fraction(
            1, 1000
        ):  # Toleranz für Rundungsfehler
            return False

        # Lösung darf kein Ausschlusswert sein
        for excl in eq.excluded:
            if abs(eq.solution - excl) < Fraction(1, 1000):
                return False

        return True
    except Exception:
        return False


def canonical_key(eq: Equation) -> str:
    """Erzeugt einen kanonischen Schlüssel für Duplikaterkennung."""
    expr = sp.expand(sp.together(eq.sympy_eq.lhs - eq.sympy_eq.rhs))
    expr = sp.collect(expr, x)
    return f"{expr}|{eq.solution}"


# ---------------------------------------------------------------------------
# Solver mit Schrittverfolgung
# ---------------------------------------------------------------------------


def solve_steps(eq: Equation, cfg: Dict) -> List[SolveStep]:
    """Erzeugt Lösungsschritte für eine Gleichung."""
    steps: List[SolveStep] = []

    # Schritt 1: Ausgangsgleichung
    steps.append(SolveStep("Ausgangsgleichung", eq.sympy_eq.lhs, eq.sympy_eq.rhs))

    # Wenn Ausschlusswerte vorhanden, erwähne sie
    if eq.excluded:
        excl_str = ", ".join([f"x ≠ {nice_frac(e)}" for e in sorted(eq.excluded)])
        steps.append(
            SolveStep(f"Definitionsmenge: {excl_str}", eq.sympy_eq.lhs, eq.sympy_eq.rhs)
        )

    # Schritt 2: Klammern auflösen
    lhs = sp.expand(eq.sympy_eq.lhs)
    rhs = sp.expand(eq.sympy_eq.rhs)

    if lhs != eq.sympy_eq.lhs or rhs != eq.sympy_eq.rhs:
        steps.append(SolveStep("Klammern auflösen", lhs, rhs))

    # Schritt 3: Brüche beseitigen (falls vorhanden)
    lhs_together = sp.together(lhs)
    rhs_together = sp.together(rhs)

    lhs_num, lhs_den = sp.fraction(lhs_together)
    rhs_num, rhs_den = sp.fraction(rhs_together)

    if lhs_den != 1 or rhs_den != 1:
        lcm = sp.lcm(lhs_den, rhs_den)

        if abs(float(lcm)) <= MAX_ABS:  # Prüfe Limit
            lhs = sp.simplify(lhs * lcm)
            rhs = sp.simplify(rhs * lcm)
            steps.append(SolveStep(f"Mit Hauptnenner {lcm} multiplizieren", lhs, rhs))

    # Schritt 4: Alle Terme auf eine Seite
    expr = sp.expand(lhs - rhs)
    steps.append(SolveStep("Alle Terme auf eine Seite bringen", expr, sp.Integer(0)))

    # Schritt 5: Nach x zusammenfassen
    expr = sp.collect(expr, x)
    steps.append(SolveStep("Nach x zusammenfassen", expr, sp.Integer(0)))

    # Schritt 6: Koeffizienten extrahieren und GCD-Normalisierung
    if expr.has(x):
        poly = sp.Poly(expr, x)
        coeffs = poly.all_coeffs()

        if len(coeffs) >= 2:
            A = int(coeffs[0])
            B = int(coeffs[1])

            # GCD-Normalisierung (NL-3)
            g = math.gcd(abs(A), abs(B))
            if g > 1:
                A //= g
                B //= g
                steps.append(
                    SolveStep(
                        f"Durch gemeinsamen Teiler {g} kürzen", A * x + B, sp.Integer(0)
                    )
                )

            # Schritt 7: Lösen
            if A != 0:
                sol = sp.Rational(-B, A)
                steps.append(SolveStep(f"Durch {A} teilen", x, sol))

                # Schritt 8: Gemischte Zahl (falls gewünscht)
                if cfg.get("solutions", {}).get("improper_and_mixed", True):
                    if sol.q != 1:  # Ist ein Bruch
                        mixed = fraction_to_mixed(Fraction(int(sol.p), int(sol.q)))
                        if mixed != str(sol):
                            steps.append(
                                SolveStep(f"Als gemischte Zahl: {mixed}", x, sol)
                            )

    return steps


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_equations(cfg: Dict) -> List[Problem]:
    """Generiert alle Gleichungen gemäß Konfiguration."""
    rng = random.Random(cfg["seed"])
    problems: List[Problem] = []
    seen_keys: Set[str] = set()
    max_resamples = cfg.get("max_resamples", MAX_RESAMPLES)
    strict = cfg.get("strict_limits", True)

    # Level 1
    for i in range(cfg["counts"]["L1"]):
        attempts = 0
        while attempts < max_resamples:
            sol = sample_solution(rng, cfg)
            eq = build_L1(rng, sol, cfg)

            limits_ok = numeric_limits_ok(eq) if strict else True
            if validate(eq) and limits_ok:
                key = canonical_key(eq)
                if key not in seen_keys:
                    seen_keys.add(key)
                    steps = solve_steps(eq, cfg)
                    problems.append(
                        Problem(
                            eq,
                            steps,
                            resamples=attempts,
                            dops=eq.params.get("dops", []),
                        )
                    )
                    break
            attempts += 1

    # Level 2
    for i in range(cfg["counts"]["L2"]):
        attempts = 0
        while attempts < max_resamples:
            sol = sample_solution(rng, cfg)
            eq = build_L2(rng, sol, cfg)

            limits_ok = numeric_limits_ok(eq) if strict else True
            if validate(eq) and limits_ok:
                key = canonical_key(eq)
                if key not in seen_keys:
                    seen_keys.add(key)
                    steps = solve_steps(eq, cfg)
                    problems.append(
                        Problem(
                            eq,
                            steps,
                            resamples=attempts,
                            dops=eq.params.get("dops", []),
                        )
                    )
                    break
            attempts += 1

    # Level 3 - genau je 2× plus, minus, times
    patterns = ["plus", "plus", "minus", "minus", "times", "times"]
    for i, pattern in enumerate(patterns[: cfg["counts"]["L3"]]):
        attempts = 0
        while attempts < max_resamples:
            sol = sample_solution(rng, cfg)
            eq = build_L3(rng, sol, cfg, i)

            limits_ok = numeric_limits_ok(eq) if strict else True
            if validate(eq) and limits_ok:
                key = canonical_key(eq)
                if key not in seen_keys:
                    seen_keys.add(key)
                    steps = solve_steps(eq, cfg)
                    problems.append(
                        Problem(
                            eq,
                            steps,
                            resamples=attempts,
                            dops=eq.params.get("dops", []),
                        )
                    )
                    break
            attempts += 1

    # Level 4
    for i in range(cfg["counts"]["L4"]):
        attempts = 0
        while attempts < max_resamples:
            sol = sample_solution(rng, cfg)
            eq = build_L4(rng, sol, cfg)

            limits_ok = numeric_limits_ok(eq) if strict else True
            if validate(eq) and limits_ok:
                key = canonical_key(eq)
                if key not in seen_keys:
                    seen_keys.add(key)
                    steps = solve_steps(eq, cfg)
                    problems.append(
                        Problem(
                            eq,
                            steps,
                            resamples=attempts,
                            dops=eq.params.get("dops", []),
                        )
                    )
                    break
            attempts += 1

    # Level 5
    for i in range(cfg["counts"]["L5"]):
        attempts = 0
        while attempts < max_resamples:
            sol = sample_solution(rng, cfg)
            eq = build_L5(rng, sol, cfg)

            limits_ok = numeric_limits_ok(eq) if strict else True
            if validate(eq) and limits_ok:
                key = canonical_key(eq)
                if key not in seen_keys:
                    seen_keys.add(key)
                    steps = solve_steps(eq, cfg)
                    problems.append(
                        Problem(
                            eq,
                            steps,
                            resamples=attempts,
                            dops=eq.params.get("dops", []),
                        )
                    )
                    break
            attempts += 1

    return problems


# ---------------------------------------------------------------------------
# DOCX Renderer
# ---------------------------------------------------------------------------


def render_arbeitsblatt(problems: List[Problem], filename: str):
    """Erstellt das Arbeitsblatt."""
    doc = Document()
    doc.add_heading("Lineare Gleichungen – Arbeitsblatt", level=1)

    # Gruppiere nach Level
    by_level = {}
    for p in problems:
        level = p.equation.level
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(p)

    # Durchgehende Nummerierung
    num = 1

    for level in ["L1", "L2", "L3", "L4", "L5"]:
        if level not in by_level:
            continue

        doc.add_heading(f"Level {level[1]}", level=2)

        for problem in by_level[level]:
            p = doc.add_paragraph(f"{num}. {problem.equation.text}")
            p.runs[0].font.size = Pt(12)

            # Leerraum zum Rechnen
            for _ in range(4):
                doc.add_paragraph("")

            num += 1

    doc.save(filename)


def render_loesungsblatt(problems: List[Problem], filename: str, cfg: Dict):
    """Erstellt das Lösungsblatt."""
    doc = Document()
    doc.add_heading("Lineare Gleichungen – Lösungsblatt", level=1)

    # Gruppiere nach Level
    by_level = {}
    for p in problems:
        level = p.equation.level
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(p)

    # Durchgehende Nummerierung
    num = 1

    for level in ["L1", "L2", "L3", "L4", "L5"]:
        if level not in by_level:
            continue

        doc.add_heading(f"Level {level[1]}", level=2)

        for problem in by_level[level]:
            doc.add_heading(f"Aufgabe {num}: {problem.equation.text}", level=3)

            # Lösungsschritte
            for step in problem.steps:
                if step.description_de.startswith("Definitionsmenge"):
                    p = doc.add_paragraph(f"• {step.description_de}")
                else:
                    eq_text = beautify_equation(
                        step.lhs, step.rhs, cfg.get("visual_complexity", "mixed")
                    )
                    p = doc.add_paragraph(f"• {step.description_de}: {eq_text}")
                p.runs[0].font.size = Pt(11)

            # Finale Lösung
            sol_frac = problem.equation.solution
            sol_str = nice_frac(sol_frac)

            if cfg.get("solutions", {}).get("improper_and_mixed", True):
                mixed = fraction_to_mixed(sol_frac)
                if mixed != sol_str:
                    p = doc.add_paragraph(f"Lösung: x = {sol_str} = {mixed}")
                else:
                    p = doc.add_paragraph(f"Lösung: x = {sol_str}")
            else:
                p = doc.add_paragraph(f"Lösung: x = {sol_str}")

            p.runs[0].bold = True
            doc.add_paragraph("")
            num += 1

    doc.save(filename)


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def main() -> None:
    """Hauptfunktion."""
    parser = argparse.ArgumentParser(description="Generator für lineare Gleichungen")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--visual", choices=["clean", "mixed"], default="mixed")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["seed"] = args.seed
    cfg["visual_complexity"] = args.visual

    print(f"Generiere Gleichungen mit Seed {cfg['seed']}...")
    problems = generate_equations(cfg)

    print(f"Erstellt: {len(problems)} Aufgaben")

    render_arbeitsblatt(problems, "arbeitsblatt.docx")
    render_loesungsblatt(problems, "loesungsblatt.docx", cfg)

    print("Dateien erstellt: arbeitsblatt.docx, loesungsblatt.docx")


if __name__ == "__main__":
    main()
