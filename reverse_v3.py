from __future__ import annotations

from typing import NamedTuple
import math


## --- Node class --- ##
# Holds each scalar in computation graph we will be building
class Node(NamedTuple):
    val: float
    parents: tuple[Node, ...] = ()
    grad_fn: callable = None


## --- Define operations on Nodes ---- ##
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


## --- Overload ops --- ##
Node.__add__ = add
Node.__sub__ = sub
Node.__mul__ = mul
Node.__truediv__ = div
Node.__pow__ = pow


## --- Functions for grad --- ##
def tree_map(f, tree):
    # taken from https://gist.github.com/okarthikb/5f3b9c8eef68bdd338f7291b27ce3df1
    if isinstance(tree, (float, int, Node)):
        return f(tree)
    elif isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(f, v) for v in tree)
    else:
        raise TypeError()


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


def contribute_grad_to_parents(
    node: Node, curr_grads: dict[int, Node]
) -> dict[int, Node]:
    """Given a node and a dictionary mapping node ids to their current grad accumulation,
    contribute the grad of the current node to its parents.

    Args:
        node: The node which will contribute grad to its parents.
        curr_grads: A dictionary mapping node ids to node's grad accumulations.

    Returns:
        dict[int, Node]: An updated dictionary mapping node ids to node's grads
        where the update is the passed-in node's contribution to its parents grads.
    """
    g = curr_grads[id(node)]
    for parent, parent_grad in zip(node.parents, node.grad_fn(g)):
        if id(parent) in curr_grads:
            curr_grads[id(parent)] += parent_grad
        else:
            curr_grads[id(parent)] = parent_grad

    return curr_grads


def value_and_grad(f: callable):

    def _value_and_grad(*args):

        in_args = tree_map(Node, args)
        out = f(*in_args)  # forward pass

        # to hold grad values of processed nodes
        grads = dict()
        grads[id(out)] = 1.0

        for node in toposort(out):

            # if we have an input node that does NOT have parents
            # it means its an input node
            if node.parents:
                contribute_grad_to_parents(node, grads)  # updates `grads` dictionary

        # gets output value and gets gradient for first input arg tree
        return out.val, tree_map(lambda n: grads[id(n)], in_args[0])

    return _value_and_grad


def grad(f: callable):

    def _grad(*args):
        _, grad = value_and_grad(f)(*args)
        return grad

    return _grad


def test():
    import jax
    import jax.numpy as jnp

    def f(inputs):
        x, y, z = inputs["x"], inputs["y"], inputs["z"]
        a = x * y
        b = x * x
        c = z * z 
        d = z - y + x
        return a + b + c + d


    our_grad = grad(f)({"x": 1.0, "y": 2.0, "z": 3.0})
    jax_grad = jax.grad(f)({"x": 1.0, "y": 2.0, "z": 3.0})

    print(f"Our grad: {our_grad}")
    print(f"Jax grad: {jax_grad}")

    # check values match
    # assert jnp.allclose(jnp.array(our_grad), jnp.array(list(jax_grad.values())))


if __name__ == "__main__":
    test()
