"""
A minimal example of forward mode auto-diff with dual numbers.
"""

class DFloat:
    def __init__(self, init_val, init_dval):
        self.val = init_val
        self.dval = init_dval

    def __repr__(self):
        return f"[{self.val}, {self.dval}]"

def without_mutating_values(init_val, init_dval):
    # // With DFloat
    def add(x: DFloat, y: DFloat):
        out_val = x.val + y.val
        out_dval = x.dval + y.dval
        return DFloat(out_val, out_dval)

    def mult(x: DFloat, y: DFloat):
        out_val = x.val * y.val
        out_dval = x.dval * y.val + x.val * y.dval
        return DFloat(out_val, out_dval)


    # // With lists
    def add_(x: list, y: list):
        out_val = x[0] + y[0]
        out_dval = x[1] + y[1]
        return [out_val, out_dval]

    def mult_(x: list, y: list):
        out_val = x[0] * y[0]
        out_dval = x[1] * y[0] + x[0] * y[1]
        return [out_val, out_dval]


    # The starting dval is like v in f'(x)v
    x = DFloat(init_val, init_dval)
    # f(x) = x * (x + x ** 2)
    print(f"With classes: {mult(x, add(x, mult(x, x)))}")

    # f(x) = x * (x + x ** 2)
    x_ = [init_val, init_dval]
    print(f"With lists: {mult_(x_, add_(x_, mult_(x_, x_)))}")


def with_mutating_values(init_val, init_dval):
    # // With DFloat
    def add(x: DFloat, y: DFloat):
        x.val = x.val + y.val
        x.dval = x.dval + y.dval
        return x
    
    def mult(x: DFloat, y: DFloat):
        x.val = x.val * y.val
        x.dval = x.dval * y.val + x.val * y.dval
        return x
    
    # // With lists
    def add_(x: list, y: list):
        x[0] = x[0] + y[0]
        x[1] = x[1] + y[1]
        return x
    
    def mult_(x: list, y: list):
        x[0] = x[0] * y[0]
        x[1] = x[1] * y[0] + x[0] * y[1]
        return x

    # The starting dval is like v in f'(x)v
    x = DFloat(init_val, init_dval)
    # f(x) = x * (x + x ** 2)
    print(f"With classes: {mult(x, add(x, mult(x, x)))}")

    # f(x) = x * (x + x ** 2)
    x_ = [init_val, init_dval]
    print(f"With lists: {mult_(x_, add_(x_, mult_(x_, x_)))}")



without_mutating_values(2.0, 1.0)
without_mutating_values(2.0, 1.0)
