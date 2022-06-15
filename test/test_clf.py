from math import cos, pi, sin
import unittest
import numpy as np
import sympy as sp
import gurobipy
from src.symbolics import evalf_expr
from src.clf import ExprTerm, Generator

class TestGenerator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        x, y = syms = sp.symbols('x y')
        exprterms = [
            ExprTerm(x**2, 2*x*(x/2 + y), [2*x*x, 0*y]),
            ExprTerm(y**2, 2*y*y/2, [0*x, 2*y*y])
        ]
        ninp = 2
        epsilon = 1e-3
        gen = Generator(syms, exprterms, ninp, epsilon)
        gen.rmax = 100

        nwit = 1000
        for a in np.linspace(0, 2*pi, nwit):
            states = np.array([cos(a), sin(a)])
            gen.add_witness(states)

        self.gen = gen

    def test_coeffs(self):
        coeffs, r = self.gen.compute_coeffs(output_flag=False)
        self.assertEqual(round(coeffs[0], 3), 0.4)
        self.assertAlmostEqual(coeffs[1], 1)
        self.assertEqual(round(r, 3), 0.2)