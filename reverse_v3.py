from __future__ import annotations

from typing import NamedTuple
import math


class Node(NamedTuple):
    val: float
    parents: tuple[Node, ...] = ()
    grad_fn: callable = None

    def __add__(self: Node, other: Node) -> Node:
        return add(self, other)

    def __sub__(self: Node, other: Node) -> Node:
        return sub(self, other)

    def __mul__(self: Node, other: Node) -> Node:
        return mul(self, other)

    def __truediv__(self: Node, other: Node) -> Node:
        return div(self, other)

    def __pow__(self: Node, other: Node) -> Node:
        return pow(self, other)


def add(
    x: Node,
    y: Node,
) -> Node:
    out = Node(val=x.val + y.val, parents=(x, y), grad_fn=lambda g: (g, g))
    return out


def sub(
    x: Node,
    y: Node,
) -> Node:
    out = Node(val=x.val - y.val, parents=(x, y), grad_fn=lambda g: (g, -g))
    return out


def mul(
    x: Node,
    y: Node,
) -> Node:
    out = Node(
        val=x.val * y.val, parents=(x, y), grad_fn=lambda g: (g * y.val, g * x.val)
    )
    return out


def div(
    x: Node,
    y: Node,
) -> Node:
    out = Node(
        val=x.val / y.val,
        parents=(x, y),
        grad_fn=lambda g: (g / y.val, -(g * x.val) / (y.val**2)),
    )
    return out


def pow(
    x: Node,
    y: Node,
) -> Node:
    out = Node(
        val=math.pow(x.val, y.val),
        parents=(x, y),
        grad_fn=lambda g: (
            g * y.value * math.pow(x.val, y.val - 1),
            g * math.pow(x.val, y.val) * math.log(x.val),
        ),
    )
    return out


def sin(
    x: Node,
) -> Node:
    out = Node(
        val=math.sin(x.val), parents=(x,), grad_fn=lambda g: (g * math.cos(x.val),)
    )
    return out


def cos(x: Node) -> Node:
    out = Node(
        val=math.cos(x.val), parents=(x,), grad_fn=lambda g: (-g * math.sin(x.val),)
    )
    return out


def toposort(node: Node):

    visited = set()
    nodes = []

    def dfs(n):
        if n not in visited:
            visited.add(n)
            for parent in n.parents:
                dfs(parent)
            nodes.append(n)

    dfs(node)
    return reversed(nodes)


def grad(f: callable):

    def _grad(*at: tuple[float, ...]):
        input_ids = {}

        out = f(*at)  # forward pass

        # to hold grad values of processed nodes
        grads = dict()
        grads[id(out)] = 1.0

        def accumulate_grad_to_parents(node: Node) -> None:

            g = grads[id(node)]
            parents = node.parents
            parents_grads = node.grad_fn(g)

            for parent, parent_grad in zip(parents, parents_grads):
                if id(parent) in grads:
                    grads[id(parent)] += parent_grad
                else:
                    grads[id(parent)] = parent_grad

        for node in toposort(out):

            # if we have an input node that doesn't have parents
            # it means its an input node
            if not node.parents:
                input_ids[id(node)] = None

            else:
                accumulate_grad_to_parents(node)

        return tuple(grads[k] for k in input_ids.keys())[::-1]

    return _grad


def build_recursive_operator(op: callable):

    def fn(*args):
        if len(args) == 1:
            return args[0]
        else:
            return op(args[0], fn(*args[1:]))

    return fn


recursive_add: callable = build_recursive_operator(add)
recursive_sub: callable = build_recursive_operator(sub)
recursive_mul: callable = build_recursive_operator(mul)
recursive_div: callable = build_recursive_operator(div)
recursive_pow: callable = build_recursive_operator(pow)


def test():
    import jax
    import jax.numpy as jnp

    x, y, z = Node(1.0), Node(2.0), Node(3.0)

    def f_jax(inputs):
        x, y, z = inputs["x"], inputs["y"], inputs["z"]
        a = x * y
        b = x * x
        c = z * z * jnp.sin(x)
        d = z - y + x
        return a + b + c + d

    def f(x, y, z):
        a = x * y
        b = x * x
        c = z * z * sin(x)
        d = z - y + x
        return recursive_add(a, b, c, d)

    our_grad = grad(f)(x, y, z)
    jax_grad = jax.grad(f_jax)({"x": 1.0, "y": 2.0, "z": 3.0})

    print(f"Our grad: {our_grad}")
    print(f"Jax grad: {jax_grad}")

    # check values match
    assert jnp.allclose(jnp.array(our_grad), jnp.array(list(jax_grad.values())))


if __name__ == "__main__":
    test()
