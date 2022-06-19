import numpy as np
from gurobipy import gurobipy, GRB, abs_
import z3
from src.symbolics import evalf_expr, diff_expr
from src.z3utils import \
    create_z3syms_from_spsyms, \
    convert_spexpr_to_z3expr, \
    get_vars_from_z3model

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

class WitnessPos:
    def __init__(self, nsates, valterms_V) -> None:
        self.nstates = nsates
        self.valterms_V = valterms_V

class WitnessLie:
    def __init__(self, nsates, valterms_DVF, valterms_DVGs) -> None:
        self.nstates = nsates
        self.valterms_DVF = valterms_DVF
        self.valterms_DVGs = valterms_DVGs

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
        self.witnesses_pos = []
        self.witnesses_lie = []

    def add_witness_pos(self, states):
        valterms_V = [
            evalf_expr(exprterm.V, self.syms, states)
            for exprterm in self.exprterms
        ]
        self.witnesses_pos.append(WitnessPos(
            np.linalg.norm(states), valterms_V
        ))
    
    def add_witness_lie(self, states):
        valterms_DVF = [
            evalf_expr(exprterm.DVF, self.syms, states)
            for exprterm in self.exprterms
        ]
        valterms_DVGs = [[
            evalf_expr(expr_DVG, self.syms, states)
            for expr_DVG in exprterm.DVGs
        ] for exprterm in self.exprterms]
        self.witnesses_lie.append(WitnessLie(
            np.linalg.norm(states), valterms_DVF, valterms_DVGs
        ))

    def compute_coeffs(self, *, output_flag=True):
        model = gurobipy.Model('Robust coeffs')
        model.setParam('OutputFlag', output_flag)
        coeffs_ = model.addVars(self.ncoeff, lb=-1, ub=+1, name='c')
        coeffs = np.array(coeffs_.values())
        r = model.addVar(lb=-float('inf'), ub=self.rmax, name='r')

        for wit in self.witnesses_pos:
            a = np.array([valterm_V for valterm_V in wit.valterms_V])
            nstates = wit.nstates
            model.addConstr(np.dot(a, coeffs) >= nstates*self.epsilon)

        for wit in self.witnesses_lie:
            nstates = wit.nstates
            con = model.addVars(self.ninp)
            for i in range(self.ninp):
                a = np.array([
                    valterm_DVGs[i] for valterm_DVGs in wit.valterms_DVGs
                ])
                z_ = model.addVar()
                model.addConstr(z_ == np.dot(a, coeffs))
                model.addConstr(con[i] == abs_(z_))
            a = np.array([valterm_DVF for valterm_DVF in wit.valterms_DVF])
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

## Verifier

class VerifierError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Verifier:
    def __init__(
            self, syms, lbs, ubs, exprterms, ninp, tol_pos, tol_lie
        ) -> None:
        assert len(syms) == len(lbs) == len(ubs)
        self.spsyms = syms
        self.lbs = lbs
        self.ubs = ubs
        self.exprterms = exprterms
        self.ninp = ninp
        self.tol_pos = tol_pos
        self.tol_lie = tol_lie

    def _check_pos_single(self, expr_V, ifix, lufix):
        ctx = z3.Context()
        solver = z3.Solver(ctx=ctx)
        z3syms, syms_map = \
            create_z3syms_from_spsyms(ctx, self.spsyms)
        # z3r = [z3.Real(f'r{i}', ctx=ctx) for i in range(self.ninp)]

        for i in range(len(z3syms)):
            if i != ifix:
                solver.add(z3syms[i] >= self.lbs[i])
                solver.add(z3syms[i] <= self.ubs[i])
            elif i == ifix:
                if lufix == -1:
                    solver.add(z3syms[i] == self.lbs[i])
                elif lufix == 1:
                    solver.add(z3syms[i] == self.ubs[i])
                else:
                    raise VerifierError(
                        'Unknow lufix = %s.' % lufix
                    )

        z3expr = convert_spexpr_to_z3expr(syms_map, expr_V)
        solver.add(z3expr <= self.tol_pos)

        res = solver.check()

        if res == z3.sat:
            model = solver.model()
            vars_ = get_vars_from_z3model(syms_map, model)
            vars = np.array([vars_[sym.name] for sym in self.spsyms])
            return False, vars
        else:
            return True, np.zeros(len(self.spsyms))

    def check_pos(self, coeffs):
        exprs_term = np.array([exprterm.V for exprterm in self.exprterms])
        expr_V = np.dot(coeffs, exprs_term)
        for ifix in range(len(self.spsyms)):
            for lufix in (-1, 1):
                res, vars = self._check_pos_single(expr_V, ifix, lufix)
                if not res:
                    return False, vars
        return True, np.zeros(len(self.spsyms))