from typing import Callable, Tuple

# // Dual number representation -----------------------------------------------

class Variable:
    def __init__(self, v: float, dv: float = 0):
        self.v = v
        self.dv = dv

    def __repr__(self):
        return f"[{self.v}, {self.dv}]"

    def __add__(self, var):
        var = self._maybe_lift(var)
        v = self.v + var.v
        dv = self.dv + var.dv
        return Variable(v, dv)

    def __mul__(self, var):
        var = self._maybe_lift(var)
        v = self.v * var.v
        dv = self.dv * var.v + self.v * var.dv
        return Variable(v, dv)

    def __sub__(self, var):  # self - var
        var = self._maybe_lift(var)
        v = self.v - var.v
        dv = self.dv - var.dv
        return Variable(v, dv)

    @staticmethod
    def _maybe_lift(var):
        if not isinstance(var, Variable):
            return Variable(var, 0)
        else:
            return var




# // Function for getting derivative with forward-mode autodiff ---------------
def value_and_grad(f: Callable, at: Tuple):

    # Create Variables, i.e. dual numbers
    n_vars = len(at)
    vars = [Variable(v) for v in at]

    value: float
    grad = list()
    for i in range(n_vars):

        # set tangent to 1 to get grad wrt to it
        vars[i].dv = 1

        # call function
        out_i = f(*vars) 

        # reset tangent to 0 for next loop
        vars[i].dv = 0

        if i == 0:
            value = out_i.v
        grad.append(out_i.dv)

    return value, grad
