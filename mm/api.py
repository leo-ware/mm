from abc import ABC, abstractmethod

class Math(ABC):
    @abstractmethod
    def _latex(self):
        raise NotImplementedError()

    def latex(self):
        l = self._latex()
        if isinstance(l, str):
            return f"${self._latex()}$"
        elif isinstance(l, tuple) or isinstance(l, list):
            return "$$" + " \\\\ ".join(l) + "$$"
        else:
            raise TypeError(f"Expected _latex to return a string or tuple, got {type(l)}")
    
    def math(self):
        try:
            from IPython.display import Math
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Need optional dependency IPython to display math widget, consider using the latex() method instead if you are in a non-interactive environment")
        return Math(self.latex())

class Effect(Math):
    def __add__(self, other):
        assert isinstance(other, Effect) or (other in [0, 1]) or isinstance(other, Variable)
        if other == 1:
            other = Intercept()
        elif other == 0:
            return self
        elif isinstance(other, Variable):
            other = Fixed(other)
        
        effects = []
        if isinstance(self, CompoundEffect):
            effects.extend(self.effects)
        else:
            effects.append(self)
        
        if isinstance(other, CompoundEffect):
            effects.extend(other.effects)
        else:
            effects.append(other)
        
        return CompoundEffect(tuple(effects))
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __or__(self, other):
        return CompoundEffect([self]).ref(other)
    
    def __ror__(self, other):
        return CompoundEffect([other]).ref(self)

class CompoundEffect(Effect):
    def __init__(self, effects):
        self.fixed_effects = []
        self.random_effects = []
        self.intercept = False

        effects = list(effects)
        while effects:
            effect = effects.pop()
            if isinstance(effect, CompoundEffect):
                effects.extend(effect.effects)
            elif isinstance(effect, Fixed):
                self.fixed_effects.append(effect)
            elif isinstance(effect, Random):
                self.random_effects.append(effect)
            elif effect == 1 or isinstance(effect, Intercept):
                self.intercept = True
            elif effect == 0:
                continue
            else:
                raise TypeError(f"Expected effect to be an Effect, got {type(effect)}")
        
        self.fixed_effects = tuple(sorted(self.fixed_effects))
        self.random_effects = tuple(sorted(self.random_effects))
    
    @property
    def effects(self):
        return ((Intercept(),) * self.intercept) + self.fixed_effects + self.random_effects
    
    def ref(self, other):
        assert len(self.effects) <= 2
        if isinstance(other, Fixed):
            other = other.name
        i = s = None
        for e in self.effects:
            if isinstance(e, Fixed):
                s = e.name
            elif isinstance(e, Intercept):
                i = True
            else:
                raise TypeError(f"RandomEffect must have only Fixed and Intercept on left hand side, got {type(e)}")
        return Random(group=varify(other), slope=s, intercept=not not i)
    
    def __repr__(self):
        return f"CompoundEffect({list(self.effects)})"

    def _latex(self):
        f_count = 0
        r_count = 0
        randoms = []
        ans = []
        for effect in self.effects:
            if isinstance(effect, Fixed):
                f_count += 1
                ans.append(effect._latex(f_count))
            elif isinstance(effect, Random):
                r_count += 1
                ans.append(effect._latex(r_count))
                if effect.slope:
                    randoms.append(effect._slope_latex(r_count) + "\\sim Normal(0, \\sigma_{\\omega," + str(r_count) + ", 1}^2)")
                if effect.intercept:
                    randoms.append(effect._slope_latex(r_count) + "\\sim Normal(0, \\sigma_{\\omega," + str(r_count) + ", 0}^2)")
            else:
                ans.append(effect._latex())

        return [" + ".join(ans), *randoms]

class Intercept(Effect):
    def __init__(self):
        pass
    
    def _latex(self):
        return "\\beta_0"
    
    def __repr__(self):
        return "Intercept()"

class Fixed(Effect):
    def __init__(self, name):
        self.name = varify(name)
    
    def __lt__(self, other):
        assert isinstance(other, Fixed)
        return self.name.name < other.name.name
    
    def __repr__(self):
        return f"Fixed({self.name})"

    def _latex(self, n=1):
        return "\\beta_{" + str(n) + "} \\text{" + self.name._latex() + "}"

class Random(Effect):
    def __init__(self, group, slope, intercept=True):
        assert isinstance(intercept, bool)
        self.group = varify(group)
        if slope is not None:
            self.slope = varify(slope)
        else:
            self.slope = None
        self.intercept = intercept
    
    def __lt__(self, other):
        assert isinstance(other, Random)
        return (self.group.name, self.slope.name, self.intercept) < (other.group.name, other.slope.name, other.intercept)
    
    def _slope_latex(self, n=1):
        return "\\omega_{\\text{" + self.group._latex() + "}}^{" + str(n) + ", 1}"
    
    def _intercept_latex(self, n=1):
        return "\\omega_{\\text{" + self.group._latex() + "}}^{" + str(n) + ", 0}"

    def _latex(self, n=1):
        ans = []
        if self.slope:
            slope_var = "\\text{" + self.slope._latex() + "}"
            ans.append(self._slope_latex(n) + slope_var)
        if self.intercept:
            ans.append(self._intercept_latex(n))
        return "+".join(ans)
    
    def __repr__(self):
        return f"Random(group={self.group}, slope={self.slope}, intercept={self.intercept})"

class Variable:
    def __init__(self, name):
        self.name = name

    def _latex(self):
        return "\\text{" + self.name + "}"
    
    def __repr__(self):
        return f"Variable({self.name})"
    
    def __add__(self, other):
        return Fixed(self) + other
    
    def __radd__(self, other):
        return self + other
    
    def __or__(self, other):
        return Fixed(self) | other
    
    def __ror__(self, other):
        return other | Fixed(self)

    def hat(self):
        return Outcome(self)

def varify(name):
    """Takes a string or Variable and returns a Variable"""
    if isinstance(name, str):
        return Variable(name)
    if isinstance(name, Variable):
        return name
    raise TypeError(f"Expected string or Variable, got {type(name)}")

def make_vars(names):
    """Takes a iterable of variable names and returns a list of Variables"""
    return [varify(name) for name in names]

class Outcome(Math):
    def __init__(self, var):
        self.var = varify(var)

    def _latex(self):
        return self.var._latex()

    def __eq__(self, other):
        assert isinstance(other, Effect)
        return Model(self, other)

class Model(Math):
    """Represents a mixed model
    
    Attributes:
        outcome: Variable
        effect: Effect
    """
    def __init__(self, outcome, effect):
        if isinstance(outcome, Outcome):
            outcome = outcome.var
        assert isinstance(outcome, Variable)
        assert isinstance(effect, Effect)
        self.outcome = outcome
        self.effect = CompoundEffect([effect])
    
    def __repr__(self):
        return f"Model({self.outcome}, {self.effect})"

    def _latex(self):
        el = self.effect._latex()
        return [
            f"{self.outcome._latex()} = {el[0]} + \\epsilon",
            "\\epsilon \\sim Normal(0, \\sigma_\\epsilon^2)",
            *el[1:]
            ]