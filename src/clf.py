import numpy as np
from gurobipy import gurobipy, GRB, abs_
import z3
from src.symbolics import evalf_expr, diff_expr
from src.z3utils import \
    create_z3syms_from_spsyms, \
    convert_spexpr_to_z3expr, \
    get_vars_from_z3model

class SysTerm:
    def __init__(self, expr_V, expr_DVF, expr_DVGs) -> None:
        self.V = expr_V
        self.DVF = expr_DVF
        self.DVGs = expr_DVGs

class SysTemplate:
    def __init__(self, systerms) -> None:
        assert bool(systerms)
        ninp = next(len(systerm.DVGs) for systerm in systerms)
        assert all(len(systerm.DVGs) == ninp for systerm in systerms)
        self.ninp = ninp
        self.systerms = systerms

class Domain:
    def __init__(self, lbs_out, ubs_out, lbs_in, ubs_in) -> None:
        nvar = len(lbs_out)
        assert all(len(b) == nvar for b in (ubs_out, lbs_in, ubs_in))
        self.nvar = nvar
        self.lbs_out = lbs_out
        self.ubs_out = ubs_out
        self.lbs_in = lbs_in
        self.ubs_in = ubs_in

## Generator

class WitnessPos:
    def __init__(self, nsates, vals_V) -> None:
        self.nstates = nsates
        self.vals_V = vals_V

class WitnessLie:
    def __init__(self, nsates, vals_DVF, vals_DVGs) -> None:
        self.nstates = nsates
        self.vals_DVF = vals_DVF
        self.vals_DVGs = vals_DVGs

class GeneratorError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Generator:
    def __init__(self, syms, systemp, epsilon) -> None:
        self.syms = syms
        self.ncoeff = len(systemp.systerms)
        self.systemp = systemp
        self.epsilon = epsilon
        self.rmax = 2
        self.witnesses_pos = []
        self.witnesses_lie = []

    def add_witness_pos(self, states):
        vals_V = [
            evalf_expr(systerm.V, self.syms, states)
            for systerm in self.systemp.systerms
        ]
        self.witnesses_pos.append(WitnessPos(
            np.linalg.norm(states), vals_V
        ))
    
    def add_witness_lie(self, states):
        vals_DVF = [
            evalf_expr(systerm.DVF, self.syms, states)
            for systerm in self.systemp.systerms
        ]
        vals_DVGs = [[
            evalf_expr(expr_DVG, self.syms, states)
            for expr_DVG in systerm.DVGs
        ] for systerm in self.systemp.systerms]
        self.witnesses_lie.append(WitnessLie(
            np.linalg.norm(states), vals_DVF, vals_DVGs
        ))

    def compute_coeffs(self, *, output_flag=True):
        model = gurobipy.Model('Robust coeffs')
        model.setParam('OutputFlag', output_flag)
        coeffs_ = model.addVars(self.ncoeff, lb=-1, ub=+1, name='c')
        coeffs = np.array(coeffs_.values())
        r = model.addVar(lb=-float('inf'), ub=self.rmax, name='r')

        for wit in self.witnesses_pos:
            a = np.array([val_V for val_V in wit.vals_V])
            nstates = wit.nstates
            if self.epsilon is None:
                model.addConstr(np.dot(a, coeffs) >= nstates*r)
            else:
                model.addConstr(np.dot(a, coeffs) >= nstates*self.epsilon)

        for wit in self.witnesses_lie:
            nstates = wit.nstates
            con = model.addVars(self.systemp.ninp)
            for i in range(self.systemp.ninp):
                a = np.array([val_DVGs[i] for val_DVGs in wit.vals_DVGs])
                z_ = model.addVar(lb=-float('inf'), ub=float('inf'))
                model.addConstr(z_ == np.dot(a, coeffs))
                model.addConstr(con[i] == abs_(z_))
            a = np.array([val_DVF for val_DVF in wit.vals_DVF])
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
    def __init__(self, syms, domain, systemp, tol_pos, tol_lie) -> None:
        assert len(syms) == domain.nvar
        self.nvar = domain.nvar
        self.spsyms = syms
        self.domain = domain
        self.systemp = systemp
        self.tol_pos = tol_pos
        self.tol_lie = tol_lie

    def _check_pos_single(self, expr_V, kfix, lufix):
        ctx = z3.Context()
        solver = z3.Solver(ctx=ctx)
        z3syms, syms_map = \
            create_z3syms_from_spsyms(ctx, self.spsyms)

        for k in range(self.nvar):
            if k != kfix:
                solver.add(z3syms[k] >= self.domain.lbs_out[k])
                solver.add(z3syms[k] <= self.domain.ubs_out[k])
            elif k == kfix:
                if lufix == -1:
                    solver.add(z3syms[k] == self.domain.lbs_out[k])
                elif lufix == 1:
                    solver.add(z3syms[k] == self.domain.ubs_out[k])
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
            return True, np.zeros(self.nvar)

    def check_pos(self, coeffs):
        exprs_V = np.array([systerm.V for systerm in self.systemp.systerms])
        expr_V = np.dot(coeffs, exprs_V)
        for kfix in range(self.nvar):
            for lufix in (-1, 1):
                res, vars = self._check_pos_single(expr_V, kfix, lufix)
                if not res:
                    return False, vars
        return True, np.zeros(self.nvar)

    def _check_lie_single(self, expr_DVF, expr_DVGs, kfix, lufix):
        ctx = z3.Context()
        solver = z3.Solver(ctx=ctx)
        z3syms, syms_map = \
            create_z3syms_from_spsyms(ctx, self.spsyms)
        z3r = [z3.Real(f'r{i}', ctx=ctx) for i in range(self.systemp.ninp)]

        for k in range(self.nvar):
            if k != kfix:
                solver.add(z3syms[k] >= self.domain.lbs_out[k])
                solver.add(z3syms[k] <= self.domain.ubs_out[k])
            elif k == kfix:
                if lufix == -1:
                    solver.add(z3syms[k] >= self.domain.lbs_out[k])
                    solver.add(z3syms[k] <= self.domain.lbs_in[k])
                elif lufix == 1:
                    solver.add(z3syms[k] >= self.domain.ubs_in[k])
                    solver.add(z3syms[k] <= self.domain.ubs_out[k])
                else:
                    raise VerifierError(
                        'Unknow lufix = %s.' % lufix
                    )

        z3expr_DV = convert_spexpr_to_z3expr(syms_map, expr_DVF)
        for i in range(self.systemp.ninp):
            z3expr_r = convert_spexpr_to_z3expr(syms_map, expr_DVGs[i])
            solver.add(z3r[i] + z3expr_r >= 0)
            solver.add(z3r[i] - z3expr_r >= 0)
            z3expr_DV = z3expr_DV - z3r[i]
        solver.add(z3expr_DV >= self.tol_lie)

        res = solver.check()

        if res == z3.sat:
            model = solver.model()
            vars_ = get_vars_from_z3model(syms_map, model)
            vars = np.array([vars_[sym.name] for sym in self.spsyms])
            return False, vars
        else:
            return True, np.zeros(self.nvar)

    def check_lie(self, coeffs):
        exprs_DVF = np.array([systerm.DVF for systerm in self.systemp.systerms])
        expr_DVF = np.dot(coeffs, exprs_DVF)
        exprs_DVGs = [
            np.array([systerm.DVGs[i] for systerm in self.systemp.systerms])
            for i in range(self.systemp.ninp)
        ]
        expr_DVGs = [np.dot(coeffs, exprs_DVG) for exprs_DVG in exprs_DVGs]
        for kfix in range(self.nvar):
            for lufix in (-1, 1):
                res, vars = self._check_lie_single(
                    expr_DVF, expr_DVGs, kfix, lufix
                )
                if not res:
                    return False, vars
        return True, np.zeros(self.nvar)

## Learner

class System:
    def __init__(self, expr_F, expr_Gs) -> None:
        self.F = expr_F
        self.Gs = expr_Gs

class Template:
    def __init__(self, exprs_V) -> None:
        self.terms = exprs_V

def _make_systemplate(syms, sys, temp):
    terms_DV = [diff_expr(expr_V, syms) for expr_V in temp.terms]
    terms_DVF = [np.dot(sys.F, term_DV) for term_DV in terms_DV]
    terms_DVGs = [
        [np.dot(expr_G, term_DV) for expr_G in sys.Gs]
        for term_DV in terms_DV
    ]
    return SysTemplate([
        SysTerm(*exprs_systerm)
        for exprs_systerm in zip(temp.terms, terms_DVF, terms_DVGs)
    ])


class LearnerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)    

class Learner:
    def __init__(
            self, syms, sys, temp, domain, epsilon, tol_rad, tol_pos, tol_lie
        ) -> None:
        assert len(syms) == domain.nvar
        self.syms = syms
        self.sys = sys
        self.systemp = _make_systemplate(syms, sys, temp)
        self.domain = domain
        self.epsilon = epsilon
        self.tol_rad = tol_rad
        self.tol_pos = tol_pos
        self.tol_lie = tol_lie
        self.iter_max = 1_000

    def learn_CLF(self):
        gen = Generator(self.syms, self.systemp, self.epsilon)
        verif = Verifier(
            self.syms, self.domain, self.systemp, self.tol_pos, self.tol_lie
        )
        iter = 0

        while True:
            iter = iter + 1
            if iter > self.iter_max:
                raise LearnerError('Max iter excedeed: ' + str(iter))

            coeffs, r = gen.compute_coeffs(output_flag=False)
            print('\nIter %5d:\n%s\n%s' % (iter, coeffs, r))

            if r < self.tol_rad:
                raise LearnerError('Radius too small: ' + str(r))

            res = True

            print('Verify pos...', end='', flush=True)
            res, states = verif.check_pos(coeffs)
            if not res:
                print(' CE found: %s' % states)
                gen.add_witness_pos(states)
                continue
            else:
                print(' No CE found')
            
            print('Verify lie...', end='', flush=True)
            res, states = verif.check_lie(coeffs)
            if not res:
                print(' CE found: %s' % states)             
                gen.add_witness_lie(states)
                continue

            print(' No CE found')
            print('Valid CLF: terminated')
            return coeffs