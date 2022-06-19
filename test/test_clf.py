from math import cos, pi, sin
import unittest
import numpy as np
import sympy as sp
from src.clf import ExprTerm, Generator, Verifier

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

        nwit_pos = 10
        for a in np.linspace(0, 2*pi, nwit_pos):
            states = np.array([cos(a), sin(a)])
            gen.add_witness_pos(states)

        nwit_lie = 1000
        for a in np.linspace(0, 2*pi, nwit_lie):
            states = np.array([cos(a), sin(a)])
            gen.add_witness_lie(states)

        self.gen = gen

    def test_coeffs(self):
        coeffs, r = self.gen.compute_coeffs(output_flag=False)
        self.assertEqual(round(coeffs[0], 3), 0.4)
        self.assertAlmostEqual(coeffs[1], 1)
        self.assertEqual(round(r, 3), 0.2)

class TestVerifier(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        x, y = syms = np.array(sp.symbols('x y'))
        lbs = np.array([-1, -1])
        ubs = np.array([1, 1])
        exprterms = [
            ExprTerm(x**2, 2*x*(x/2 + y), [2*x*x, 0*y]),
            ExprTerm(y**2, 2*y*y/2, [0*x, 2*y*y])
        ]
        ninp = 2
        tol_pos = 0.1
        tol_lie = 0.1

        self.verif = Verifier(syms, lbs, ubs, exprterms, ninp, tol_pos, tol_lie)

    def test_check_pos(self):
        coeffs = np.array([1, 0.1])
        res, vars = self.verif.check_pos(coeffs)
        self.assertFalse(res)
        self.assertAlmostEqual(abs(vars[1]), 1)
        coeffs = np.array([1, 0.1 + 1e5])
        res, vars = self.verif.check_pos(coeffs)
        self.assertTrue(res)