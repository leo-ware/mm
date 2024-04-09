from mm.infix_api import *

def test_variable():
    x, y, z, w = make_vars("xyzw")
    eq1 = y.hat() == (1 | x) + z
    assert isinstance(eq1, ModelEquation)
    assert (1 | x) in eq1.effect
    assert z in eq1.effect
    assert x in eq1.effect
    assert 1 not in eq1.effect
    assert w not in eq1.effect

    assert isinstance(eq1.vars, frozenset)
    assert isinstance(eq1.effect.vars, frozenset)
    assert isinstance((1 | x).vars, frozenset)

def test_random_effect():
    x, y, z, w = make_vars("xyzw")
    re1 = (x + 1 | z)
    assert isinstance(re1, RandomEffect)
    # assert x in re1
    # assert z in re1
    assert re1.group is z
    assert re1.intercept is True
    assert re1.slope is x
