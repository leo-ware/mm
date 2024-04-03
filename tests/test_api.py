from src.api import make_vars, Variable, Outcome, Model

def test_variable():
    x, y, z = make_vars("xyz")
    eq1 = y.hat() == (1 | x) + z
    assert isinstance(eq1, Model)