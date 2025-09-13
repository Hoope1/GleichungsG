import sympy as sp
from fractions import Fraction

from gleichungs_generator import Equation, numeric_limits_ok, x


def make_eq(lhs, rhs, sol=Fraction(0)):
    return Equation("T", sp.Eq(lhs, rhs, evaluate=False), "", sol)


def test_numeric_limits_ok_pass():
    eq = make_eq(sp.Rational(1, 3) * x, sp.Rational(2, 5), Fraction(10, 3))
    assert numeric_limits_ok(eq)


def test_numeric_limits_lcm_fail():
    eq = make_eq(sp.Rational(1, 121) * x, 0, Fraction(0))
    assert not numeric_limits_ok(eq)


def test_numeric_limits_coeff_fail():
    eq = make_eq(121 * x + 1, 0, Fraction(-1, 121))
    assert not numeric_limits_ok(eq)


def test_numeric_limits_symbolic_denominator():
    eq = make_eq(1 / (x + 2), 0, Fraction(0))
    assert numeric_limits_ok(eq)
