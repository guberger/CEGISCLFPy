import numpy as np
from gurobipy import gurobipy, GRB, abs_
from src.symbolics import evalf_expr

class ExprTerm:
    def __init__(self, expr_V, expr_DVF, expr_DVGs) -> None:
        self.V = expr_V
        self.DVF = expr_DVF
        self.DVGs = expr_DVGs

class ValTerm:
    def __init__(self, val_V, val_DVF, val_DVGs) -> None:
        self.V = val_V
        self.DVF = val_DVF
        self.DVGs = val_DVGs

def _make_valterm(syms, exprterm, states):
    val_V = evalf_expr(exprterm.V, syms, states)
    val_DVF = evalf_expr(exprterm.DVF, syms, states)
    val_DVGs = [
        evalf_expr(expr_DVG, syms, states) for expr_DVG in exprterm.DVGs
    ]
    return ValTerm(val_V, val_DVF, val_DVGs)

## Generator

class Witness:
    def __init__(self, nsates, valterms) -> None:
        self.nstates = nsates
        self.valterms = valterms

class GeneratorError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Generator:
    def __init__(self, syms, exprterms, ninp, epsilon) -> None:
        self.syms = syms
        ncoeff = len(exprterms)
        self.ncoeff = ncoeff
        self.exprterms = exprterms
        self.ninp = ninp
        self.epsilon = epsilon
        self.rmax = 2
        self.witnesses = []

    def add_witness(self, states):
        valterms = [
            _make_valterm(self.syms, exprterm, states)
            for exprterm in self.exprterms
        ]
        self.witnesses.append(Witness(np.linalg.norm(states), valterms))

    def compute_coeffs(self, *, output_flag=True):
        model = gurobipy.Model('Robust coeffs')
        model.setParam('OutputFlag', output_flag)
        coeffs_ = model.addVars(self.ncoeff, lb=-1, ub=+1, name='c')
        coeffs = np.array(coeffs_.values())
        r = model.addVar(lb=-float('inf'), ub=self.rmax, name='r')

        for wit in self.witnesses:
            a = np.array([valterm.V for valterm in wit.valterms])
            nstates = wit.nstates
            model.addConstr(np.dot(a, coeffs) >= nstates*self.epsilon)
            con = model.addVars(self.ninp)
            for i in range(self.ninp):
                a = np.array([valterm.DVGs[i] for valterm in wit.valterms])
                z_ = model.addVar()
                model.addConstr(z_ == np.dot(a, coeffs))
                model.addConstr(con[i] == abs_(z_))
            a = np.array([valterm.DVF for valterm in wit.valterms])
            model.addConstr(np.dot(a, coeffs) - con.sum() + nstates*r <= 0)

        model.setObjective(r, GRB.MAXIMIZE)
        model.optimize()

        if model.Status != 2:
            raise GeneratorError(
                'Chebyshev center status = %d.' % model.status
            )

        coeffs_opt = np.array([coeff.X for coeff in coeffs])
        r_opt = model.getObjective().getValue()
        return coeffs_opt, r_opt