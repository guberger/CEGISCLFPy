from math import cos, pi, sin
import unittest
import numpy as np
import sympy as sp
from src.clf import \
    SysTerm, SysTemplate, Domain, Generator, Verifier, \
    System, Template, Learner, LearnerError

"""
F = [0.5 1; 0 0.5] : [x/2 + y, y/2]
G1 = [-1 0; 0 0] : [-x, 0]
G2 = [0 0; 0 1] : [0, y]
V = a1*x^2 + a2*y^2
"""

class TestGenerator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        x, y = syms = sp.symbols('x y')
        systemp = SysTemplate([
            SysTerm(x**2, 2*x*(x/2 + y), [2*x*x, sp.sympify(0)]),
            SysTerm(y**2, 2*y*y/2, [sp.sympify(0), 2*y*y])
        ])
        epsilon = 1e-3
        gen = Generator(syms, systemp, epsilon)
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
        domain = Domain(
            lbs_out=np.array([-1, -1]),
            ubs_out=np.array([1, 1]),
            lbs_in=np.array([-0.1, -0.1]),
            ubs_in=np.array([0.1, 0.1])
        )
        systemp = SysTemplate([
            SysTerm(x**2, 2*x*(x/2 + y), [-2*x*x, 0*y]),
            SysTerm(y**2, 2*y*y/2, [0*x, 2*y*y])
        ])
        tol_pos = 0.1
        tol_lie = 0.0

        self.verif = Verifier(syms, domain, systemp, tol_pos, tol_lie)

    def test_check_pos(self):
        coeffs = np.array([1, 0.1])
        res, vars = self.verif.check_pos(coeffs)
        self.assertFalse(res)
        self.assertAlmostEqual(abs(vars[1]), 1)
        coeffs = np.array([1, 0.1 + 1e5])
        res, vars = self.verif.check_pos(coeffs)
        self.assertTrue(res)
        self.assertAlmostEqual(abs(vars[0]) + abs(vars[1]), 0)

    def test_check_lie(self):
        coeffs = np.array([0.5, 1])
        res, vars = self.verif.check_lie(coeffs)
        self.assertTrue(res)
        self.assertAlmostEqual(abs(vars[0]) + abs(vars[1]), 0)
        coeffs = np.array([1, 1])
        res, vars = self.verif.check_lie(coeffs)
        self.assertFalse(res)
        self.assertGreaterEqual(abs(vars[0]) + abs(vars[1]), 0.1)
        self.assertGreaterEqual(vars[0], -1)
        self.assertGreaterEqual(vars[1], -1)
        self.assertLessEqual(vars[0], 1)
        self.assertLessEqual(vars[1], 1)

class TestLearner(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def _test_small(self):
        x, y = syms = np.array(sp.symbols('x y'))
        sys = System(
            expr_F=np.array([-x, 2*x/pi]),
            expr_Gs=[
                np.array([-x/10, sp.sympify(0)]),
                np.array([sp.sympify(0), sp.sympify(2)])
            ]
        )
        temp = Template([x, y, x**2, x*y, y**2])
        domain = Domain(
            lbs_out=np.array([-pi/2, -1]),
            ubs_out=np.array([pi/2, 1]),
            lbs_in=np.array([-0.1, -0.1]),
            ubs_in=np.array([0.1, 0.1])
        )
        epsilon = None
        tol_rad = 1e-5
        tol_pos = 1e-4
        tol_lie = -0.001

        lear = Learner(
            syms, sys, temp, domain, epsilon, tol_rad, tol_pos, tol_lie
        )

        lear.learn_CLF()

    def _test_medium(self):
        x1, x2, x3, x4 = syms = np.array(sp.symbols('x1 x2 x3 x4'))
        sys = System(
            expr_F=np.array([-x1, -x2, -x3, 2*x1/pi]),
            expr_Gs=[
                np.array([
                    sp.sympify(0), sp.sympify(0), sp.sympify(0), sp.sympify(2)
                ]),
                np.array([
                    -x1/100, sp.sympify(0), sp.sympify(0), sp.sympify(0)
                ])
            ]
        )
        temp = Template([
            x1*x2, x1*x3, x1*x4, x2*x3, x3*x4,
            x1**2, x2**2, x3**2, x4**2]) # works not with one extra xi*xj...
        domain = Domain(
            lbs_out=np.array([-pi/2, -1, -1, -1]),
            ubs_out=np.array([pi/2, 1, 1, 1]),
            lbs_in=np.array([-0.1, -0.1, -0.1, -0.1]),
            ubs_in=np.array([0.1, 0.1, 0.1, 0.1])
        )
        epsilon = None
        tol_rad = 1e-5
        tol_pos = 1e-4
        tol_lie = -0.001

        lear = Learner(
            syms, sys, temp, domain, epsilon, tol_rad, tol_pos, tol_lie
        )

        coeffs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 0])
        verif = Verifier(
            lear.syms, lear.domain, lear.systemp, lear.tol_pos, lear.tol_lie
        )
        res, states = verif.check_lie(coeffs)
        print(res)
        print(states)

        lear.learn_CLF()

    def test_ackermann(self):
        x1, x2, x3, x4 = syms = np.array(sp.symbols('x1 x2 x3 x4'))
        sys = System(
            expr_F=np.array([
                x3*(1 - (x4**2)/2),
                x3*(x4 - (x4**3)/6),
                sp.sympify(0),
                sp.sympify(0)
            ]),
            expr_Gs=[
                np.array([
                    sp.sympify(0), sp.sympify(0), sp.sympify(1), sp.sympify(0)
                ]),
                np.array([
                    sp.sympify(0), sp.sympify(0), sp.sympify(0), x3/0.33
                ])
            ]
        )
        temp = Template([
            x1*x3, x2*x3,
            x1**2, x2**2, x3**2, x4**2]) # works not with one extra xi*xj...
        domain = Domain(
            lbs_out=np.array([-1, -1, -1, -pi/4]),
            ubs_out=np.array([-1, 1, 1, pi/4]),
            lbs_in=np.array([-0.2, -0.2, -0.2, -0.2*pi/4]),
            ubs_in=np.array([0.2, 0.2, 0.2, 0.2*pi/4])
        )
        epsilon = None
        tol_rad = 1e-5
        tol_pos = 1e-4
        tol_lie = -0.001

        lear = Learner(
            syms, sys, temp, domain, epsilon, tol_rad, tol_pos, tol_lie
        )

        lear.learn_CLF()