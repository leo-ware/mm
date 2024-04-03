from mm.api import make_vars, Variable, _Outcome, Model

def test_variable():
    x, y, z, w = make_vars("xyzw")
    eq1 = y.hat() == (1 | x) + z
    assert isinstance(eq1, Model)
    assert (1 | x) in eq1.effect
    assert z in eq1.effect
    assert x in eq1.effect
    assert 1 not in eq1.effect
    assert w not in eq1.effect

    assert isinstance(eq1.vars, frozenset)
    assert isinstance(eq1.effect.vars, frozenset)
    assert isinstance((1 | x).vars, frozenset)
