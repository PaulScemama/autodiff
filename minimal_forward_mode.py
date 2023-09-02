"""
A minimal example of forward mode auto-diff with dual numbers. 

Only able to compute functions with addition and multiplication.

Only able to compute functions with 1 or 2 input variables and 1 output variable.
"""

class DFloat:
    def __init__(self, v, dv):
        self.v = v
        self.dv = dv
    def __repr__(self):
        return f"[{self.v}, {self.dv}]"
    
def add(x1: DFloat, x2: DFloat):
        v = x1.v + x2.v
        dv = x1.dv + x2.dv
        return DFloat(v, dv)

def mult(x1: DFloat, x2: DFloat):
    v = x1.v * x2.v
    dv = x1.dv * x2.v + x1.v * x2.dv
    return DFloat(v, dv)



# f(x1, x2) = x1 * x2 + x1
print("\n f(x1, x2) = x1 * x2 + x1")
# Get df/dx1 evaluated at (1, 2)
x1_ = DFloat(1, 1) # set primal to 1; tangent to 1
x2_ = DFloat(2, 0) # set primal to 2; tangent to 0

# Evaluate f with x1_ and x2_
print(f"\n ----- Evaluate f(1, 2) and simultaneously compute df/dx1 ------")
i_11 = mult(x1_, x2_)
print(f"i_11: {i_11}")
o = add(i_11, x1_)
print(f"o: {o}")

# // --------------------------------------------------

# Get df/dx2 evaluated at (1, 2)
x1_ = DFloat(1, 0) # set tangent to 0
x2_ = DFloat(2, 1) # set tangent to 1

# Evaluate f with x1_ and x2_
print(f"\n ------ Evaluate f(1, 2) and simultaneously compute df/dx2 ------ ")
i_21 = mult(x1_, x2_)
print(f"i_21: {i_21}")
o = add(i_21, x1_)
print(f"o: {o}")


