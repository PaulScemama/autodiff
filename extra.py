from reverse_v3 import add, sub, mul, div, pow

## --- Extra utility for recursive operators --- ##
def build_recursive_operator(op: callable):

    def fn(*args):
        if len(args) == 1:
            return args[0]
        else:
            return op(args[0], fn(*args[1:]))

    return fn


radd: callable = build_recursive_operator(add)
rsub: callable = build_recursive_operator(sub)
rmul: callable = build_recursive_operator(mul)
rdiv: callable = build_recursive_operator(div)
rpow: callable = build_recursive_operator(pow)