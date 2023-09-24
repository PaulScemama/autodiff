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
