from __future__ import annotations

from typing import NamedTuple

"""
Forward pass procedure: for each operation create new node that holds...
    - its value
    - parents that were inputs to the operation that created the new node
    - grad function that takes in a value 'g' and return the derivative of 
      the operation wrt to each parent times that value 'g'.

NOTE: simplified with only one output node
Backward pass procedure: for each node in toposort (dfs) order
    - if it has parents, calculate the contribution of the node to
      the grad of its parents using its grad_fn and accumulate the
      result to the parents values in the grads dict.
"""


class Node(NamedTuple):
    val: float
    parents: tuple[Node, ...] = ()
    grad_fn: callable = None


def _add(
    x: Node,
    y: Node,
) -> Node:
    new_node = Node(val=x.val + y.val, parents=(x, y), grad_fn=lambda g: (g, g))
    return new_node


def _sub(
    x: Node,
    y: Node,
) -> Node:
    new_node = (Node(val=x.val - y.val, parents=(x, y), grad_fn=lambda g: (g, -g)),)
    return new_node


def _mult(
    x: Node,
    y: Node,
) -> Node:
    new_node = Node(
        val=x.val * y.val, parents=(x, y), grad_fn=lambda g: (g * y.val, g * x.val)
    )
    return new_node


def sum(*args):
    if len(args) == 1:
        return args[0]
    else:
        return _add(args[0], sum(*args[1:]))


def mult(*args):
    if len(args) == 1:
        return args[0]
    else:
        return _mult(args[0], mult(*args[1:]))


def sub(*args):
    if len(args) == 1:
        return args[0]
    else:
        return _sub(args[0], sub(*args[1:]))


def toposort(node: Node):
    """Topologically sort graph that 'created' the input nodes using
    depth-first search."""
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


def grad(f: callable, at: tuple[float, ...]):
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


if __name__ == "__main__":
    import jax

    def f_jax(inputs):
        x, y, z = inputs["x"], inputs["y"], inputs["z"]
        return x * y + x * x + z * z

    grads_jax = jax.grad(f_jax)({"x": 1.0, "y": 2.0, "z": 3.0})

    def f(x, y, z):
        return _add(_add(mult(x, y), mult(x, x)), mult(z, z))

    grads = grad(f, at=(Node(1.0), Node(2.0), Node(3.0)))

    print(grads)
    print(grads_jax)
