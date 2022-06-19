import z3
import sympy as sp
import numpy as np

class Z3UtilsError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def create_z3syms_from_spsyms(ctx, spsyms):
    z3syms_ = []
    syms_map = {}
    for spsym in spsyms:
        name = spsym.name
        z3sym = z3.Real(name, ctx=ctx)
        z3syms_.append(z3sym)
        syms_map[name] = z3sym
    return np.array(z3syms_), syms_map

def convert_spexpr_to_z3expr(syms_map, spexpr):
    z3expr = None

    if not isinstance(spexpr, sp.core.Expr):
        raise RuntimeError('Expected SymPy Expr: ' + repr(spexpr))

    if isinstance(spexpr, sp.core.Symbol):
        z3expr = syms_map.get(spexpr.name)
        if z3expr == None:
            raise Z3UtilsError(
                'No Z3 symbol corresponds to SymPy symbol: ' + \
                repr(spexpr)
            )
    elif isinstance(spexpr, sp.core.Number):
        z3expr = float(spexpr)
    elif isinstance(spexpr, sp.core.Mul):
        z3expr = convert_spexpr_to_z3expr(syms_map, spexpr.args[0])
        for child in spexpr.args[1:]:
            z3expr *= convert_spexpr_to_z3expr(syms_map, child)
    elif isinstance(spexpr, sp.core.Add):
        z3expr = convert_spexpr_to_z3expr(syms_map, spexpr.args[0])
        for child in spexpr.args[1:]:
            z3expr += convert_spexpr_to_z3expr(syms_map, child)
    elif isinstance(spexpr, sp.core.Pow):
        base = convert_spexpr_to_z3expr(syms_map, spexpr.args[0])
        exponent = spexpr.args[1]
        if isinstance(exponent, sp.core.Integer):
            z3expr = base**int(exponent)
        else:
            raise Z3UtilsError('Exponent not integer: ' + repr(exponent))

    if z3expr == None:
        raise Z3UtilsError('SymPy expr not recognized: ' + repr(spexpr))

    return z3expr

def _parse_z3expr(z3expr):
    if isinstance(z3expr, z3.RatNumRef):
        return z3expr.numerator().as_long()/z3expr.denominator().as_long()
    else:
        raise Z3UtilsError('Parse not defined for this type.')

def get_vars_from_z3model(syms_map, model):
    return {
        name:_parse_z3expr(model[z3sym])
        for name, z3sym in syms_map.items()
    }