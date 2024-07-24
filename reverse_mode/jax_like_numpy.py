from __future__ import annotations

from typing import NamedTuple, Generator, Callable, Any
import math
import numpy as np

"""
    x       y   # parents
     \     /
      \   /
       out      # node
"""


## --- Node class --- ##
# Holds each scalar in computation graph we will be building
class Node(NamedTuple):
    val: np.array
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


def dot(x: Node, y: Node) -> Node:
    out = Node(
        val=np.dot(x.val, y.val), parents=(x,y), grad_fn=lambda g: (g * y.val, g * x.val),
    )
    return out

    
# TODO: understand this from https://github.com/mattjj/autodidact/blob/master/autograd/numpy/numpy_vjps.py
# def _dot_grad_0(g: float, xval: float, yval: float) -> tuple:
    

# def _dot_grad_1(g: float, xval: float, yval: float) -> tuple:
#     if yval.ndim == 0:
#         return np.sum(xval * g)
#     if xval.ndim == 1 and yval.ndim == 1:
#         return g * xval



## --- Overload ops --- ##
Node.__add__ = add
Node.__sub__ = sub
Node.__mul__ = mul
Node.__truediv__ = div
Node.__pow__ = pow


## --- Functions for grad --- ##
def tree_map(f: Callable, tree: Any) -> Any:
    # taken from https://gist.github.com/okarthikb/5f3b9c8eef68bdd338f7291b27ce3df1
    if isinstance(tree, (np.ndarray, Node)):
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
        if id(n) not in visited:
            visited.add(id(n))
            for parent in n.parents:
                dfs(parent)
            nodes.append(n)

    dfs(node)
    return reversed(nodes)


def grad(f: Callable) -> Callable[..., tuple[float, tuple[float, ...]]]:

    def _grad(*args):

        in_args: tuple[Node, ...] = tree_map(Node, args)
        out: Node = f(*in_args)  # forward pass

        # to hold grad values of processed nodes
        grads = dict()
        grads[id(out)] = 1.0

        toposorted: Generator = toposort(out)

        for node in toposorted:
            
            # if we have an input node that does NOT have parents
            # it means its an input node
            if not node.parents:
                continue

            parents: tuple[Node, ...] = node.parents
            parents_grads: tuple[float, ...] = node.grad_fn(grads[id(node)])

            for parent, parent_grad in zip(parents, parents_grads):
                if id(parent) in grads:
                    grads[id(parent)] = grads[id(parent)] + parent_grad
                else:
                    grads[id(parent)] = parent_grad

        # gets gradient for first input arg tree
        return tree_map(lambda n: grads[id(n)], in_args[0])

    return _grad


def test():

    import jax
    import jax.numpy as jnp
    
    def f(inputs):
        x, y, z = inputs["x"], inputs["y"], inputs["z"]
        a = x * y
        b = x * x
        c = z * z
        d = y + x
        out = dot(a, b) + dot(c, d)
        return out
    

    def f_jax(inputs):
        x, y, z = inputs["x"], inputs["y"], inputs["z"]
        a = x * y
        b = x * x
        c = z * z
        d = y + x
        out = jnp.dot(a, b) + jnp.dot(c, d)
        return out

    inputs = {"x": np.array([1.0, 3.0]), "y": np.array([2.0, 4.0]), "z": np.array([3.0, 5.0])}
    our_grad = grad(f)(inputs)
    jax_grad = jax.grad(f_jax)(inputs)

    

    print(f"Our grad: {our_grad}")
    print(f"Jax grads: {jax_grad}")



if __name__ == "__main__":
    test()
