import unittest
import sympy as sp
import numpy as np
import z3
from src.symbolics import evalf_expr
from src.z3utils import \
    create_z3syms_from_spsyms, \
    convert_spexpr_to_z3expr, \
    get_vars_from_z3model

class TestZ3utils(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        spsyms = np.array(sp.symbols('x y'))
        ctx = z3.Context()
        solver = z3.Solver(ctx=ctx)
        _, syms_map = create_z3syms_from_spsyms(ctx, spsyms)
        x, y = spsyms
        spexpr1 = x**2 + 3.14159*x*y + y**2
        z3expr1 = convert_spexpr_to_z3expr(syms_map, spexpr1)
        spexpr2 = 0.1*x + 0.2*y
        z3expr2 = convert_spexpr_to_z3expr(syms_map, spexpr2)
        solver.add(z3expr1 <= -1)
        solver.add(z3expr2 >= 0)
        self.spsyms = spsyms
        self.syms_map = syms_map
        self.spexpr1 = spexpr1
        self.spexpr2 = spexpr2
        self.solver = solver

    def test_z3utils(self):
        res = self.solver.check()
        self.assertEqual(res, z3.sat)
        model = self.solver.model()
        vars_ = get_vars_from_z3model(self.syms_map, model)
        vars = np.array([vars_[sym.name] for sym in self.spsyms])
        self.assertLessEqual(evalf_expr(self.spexpr1, self.spsyms, vars), -1)
        self.assertGreaterEqual(evalf_expr(self.spexpr2, self.spsyms, vars), 0)