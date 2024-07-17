from __future__ import annotations


from typing import Generator, Callable
import math

"""
                out       # node
               /   \
              /     \
             x       y    # children
"""


## --- Node class --- ##
# Holds each scalar in computation graph we will be building
class Node:

    def __init__(
        self, val: float, children: tuple[Node, ...] = (), grad_fn: Callable = None
    ) -> None:
        self.val = val
        self.children = children
        self.grad_fn = grad_fn
        self.grad = None

    def toposort(self) -> Generator:
        visited = set()
        nodes = []

        def dfs(n):
            if n not in visited:
                visited.add(n)
                for child in n.children:
                    dfs(child)
                nodes.append(n)

        dfs(self)
        return reversed(nodes)

    def backward(self) -> None:
        self.grad = 1.0

        for node in self.toposort():

            if node.children:
                children: tuple[Node, ...] = node.children
                child_grads: tuple[float, ...] = node.grad_fn(node.grad)

                for child, child_grad in zip(children, child_grads):
                    if child.grad:
                        child.grad += child_grad
                    else:
                        child.grad = child_grad


## --- Define operations on Nodes ---- ##
def add(
    x: Node,
    y: Node,
) -> Node:
    out = Node(val=x.val + y.val, children=(x, y), grad_fn=lambda g: (g, g))
    return out


def sub(
    x: Node,
    y: Node,
) -> Node:
    out = Node(val=x.val - y.val, children=(x, y), grad_fn=lambda g: (g, -g))
    return out


def mul(
    x: Node,
    y: Node,
) -> Node:
    out = Node(
        val=x.val * y.val, children=(x, y), grad_fn=lambda g: (g * y.val, g * x.val)
    )
    return out


def div(
    x: Node,
    y: Node,
) -> Node:
    out = Node(
        val=x.val / y.val,
        children=(x, y),
        grad_fn=lambda g: (g / y.val, -(g * x.val) / (y.val**2)),
    )
    return out


def pow(
    x: Node,
    y: Node,
) -> Node:
    out = Node(
        val=math.pow(x.val, y.val),
        children=(x, y),
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
        val=math.sin(x.val), children=(x,), grad_fn=lambda g: (g * math.cos(x.val),)
    )
    return out


def cos(x: Node) -> Node:
    out = Node(
        val=math.cos(x.val), children=(x,), grad_fn=lambda g: (-g * math.sin(x.val),)
    )
    return out


## --- Overload ops --- ##
Node.__add__ = add
Node.__sub__ = sub
Node.__mul__ = mul
Node.__truediv__ = div
Node.__pow__ = pow


def test():
    import torch
    from torch import tensor

    def f(inputs):
        x, y, z = inputs["x"], inputs["y"], inputs["z"]
        a = x * y
        b = x * x
        c = z * z
        d = z - y + x
        return a + b + c + d

    our_inputs = {"x": Node(1.0), "y": Node(2.0), "z": Node(3.0)}
    our_out = f(our_inputs)
    our_out.backward()

    torch_inputs = {
        "x": tensor([1.0], requires_grad=True),
        "y": tensor([2.0], requires_grad=True),
        "z": tensor([3.0], requires_grad=True),
    }
    torch_out = f(torch_inputs)
    torch_out.backward()

    our_grad = {k: v.grad for k, v in our_inputs.items()}
    torch_grad = {k: v.grad for k, v in torch_inputs.items()}

    print(f"Our grad: {our_grad}")
    print(f"Torch grad: {torch_grad}")

    # check values match
    assert torch.allclose(
        torch.tensor(list(our_grad.values())), torch.tensor(list(torch_grad.values()))
    )


if __name__ == "__main__":
    test()
