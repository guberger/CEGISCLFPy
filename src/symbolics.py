import numpy as np
import sympy as sp

def evalf_expr(expr, syms, vars):
    valuation = {sym:var for sym, var in zip(syms, vars)}
    return float(expr.evalf(subs=valuation))

def diff_expr(expr, syms):
    return np.array([sp.diff(expr, sym) for sym in syms])