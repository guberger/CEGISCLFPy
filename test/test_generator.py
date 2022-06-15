from math import sqrt
import unittest
import numpy as np
from gurobipy import gurobipy, GRB

class GeneratorError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class TestChebyshev(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        gurobipy.Model('')
        self.rmax = 100
        
    def _test_chebyshev(self, nvar):
        model = gurobipy.Model('Robust coeffs')
        model.setParam('OutputFlag', 0)
        x_ = model.addVars(nvar, lb=-2, ub=2, name='x')
        x = np.array(x_.values())
        r = model.addVar(lb=-float('inf'), ub=self.rmax, name='r')

        for i in range(nvar):
            a = np.array([1. if j == i else 0. for j in range(nvar)])
            na = np.linalg.norm(a)
            model.addConstr(np.dot(a, x) + na*r <= 1)
            model.addConstr(np.dot(a, x) - na*r >= 0)

        a = np.ones(nvar)
        na = np.linalg.norm(a)
        model.addConstr(np.dot(a, x) + na*r <= 1)

        model.setObjective(r, GRB.MAXIMIZE)
        model.optimize()

        if model.Status != 2:
            raise GeneratorError(
                'Chebyshev center status = %d.' % model.status
            )

        xopt = np.array([var.X for var in x])
        ropt = model.getObjective().getValue()

        self.assertAlmostEqual(ropt, sqrt(1/nvar)/(1 + sqrt(nvar)))
        self.assertAlmostEqual(sum(xopt), ropt*nvar)

    def test_chebyshev(self):
        for nvar in range(1, 11):
            self._test_chebyshev(nvar)