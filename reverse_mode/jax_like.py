from __future__ import annotations

from typing import NamedTuple, Generator, Callable, Any
import math


"""
    x       y   # parents
     \     /
      \   /
       out      # node
"""


## --- Node class --- ##
# Holds each scalar in computation graph we will be building
class Node(NamedTuple):
    val: float
    parents: tuple[Node, ...] = ()
    grad_fn: Callable = None


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
def tree_map(f: Callable, tree: Any) -> Any:
    # taken from https://gist.github.com/okarthikb/5f3b9c8eef68bdd338f7291b27ce3df1
    if isinstance(tree, (float, int, Node)):
        return f(tree)
    elif isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(f, v) for v in tree)
    else:
        raise TypeError()


def toposort(node: Node) -> Generator:

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


def value_and_grad(f: Callable) -> Callable[..., tuple[float, tuple[float, ...]]]:

    def _value_and_grad(*args):

        in_args: tuple[Node, ...] = tree_map(Node, args)
        out: Node = f(*in_args)  # forward pass

        # to hold grad values of processed nodes
        grads = dict()
        grads[out] = 1.0

        toposorted: Generator = toposort(out)
        for node in toposorted:

            # if we have an input node that does NOT have parents
            # it means its an input node
            if not node.parents:
                continue

            parents: tuple[Node, ...] = node.parents
            parents_grads: tuple[float, ...] = node.grad_fn(grads[node])

            for parent, parent_grad in zip(parents, parents_grads):
                if parent in grads:
                    grads[parent] += parent_grad
                else:
                    grads[parent] = parent_grad

        # gets output value and gets gradient for first input arg tree
        out_val: float = out.val
        in_grads: tuple[float, ...] = tree_map(lambda n: grads[n], in_args[0])
        return out_val, in_grads

    return _value_and_grad


def grad(f: Callable) -> Callable[..., tuple[float, ...]]:

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
    assert jnp.allclose(
        jnp.array(list(our_grad.values())), jnp.array(list(jax_grad.values()))
    )


if __name__ == "__main__":
    test()
