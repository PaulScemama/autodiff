from __future__ import annotations

from typing import NamedTuple


"""
Forward pass procedure: for each operation
- create new node to hold...
    - output float of operation
    - parents that were inputs in creating the operation
    - vjp function that takes in a value 'g' and return the derivative of 
    the operation wrt to each parent times that value 'g'.


Backward pass procedure: for each output node (NOTE: simplified with only one output node)
- create a dictionary mapping a unique id for each node in the computation graph
- the output node should start with an grad value of 1 in that dictionary
- get a topological sorting of computation graph
- loop through nodes in sorting and accumulate gradients by...
    - get parents of node, compute parent grads using the vjp of the node
    - add parent grads to correct value of id in parent grads dict
- return leaf nodes in gradient accumulation (those without parents)
"""


class Node(NamedTuple):
    val: float
    parents: tuple[Node, Node] = ()
    vjp: callable = None
    name: str = None


def add(x: Node, y: Node, out_name: str = None) -> Node:
    new_node = Node(
        val=x.val + y.val,
        parents=(x, y),
        vjp=lambda g: (g, g),
        name=out_name,
    )
    return new_node


def subtract(x: Node, y: Node, out_name: str = None) -> Node:
    new_node = (
        Node(
            val=x.val - y.val,
            parents=(x, y),
            vjp=lambda g: (g, -g),
            name=out_name,
        ),
    )
    return new_node


def mult(x: Node, y: Node, out_name: str = None) -> Node:
    new_node = Node(
        val=x.val * y.val,
        parents=(x, y),
        vjp=lambda g: (g * y.val, g * x.val),
        name=out_name,
    )
    return new_node


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
    return nodes[::-1]


if __name__ == "__main__":

    def f(x, y):
        z1 = mult(x, y, out_name="z1")
        z2 = mult(x, x, out_name="z2")
        o = add(z1, z2, out_name="o")
        return o

    def grad(f, at):

        out = f(*at)  # forward pass
        print(out.val)

        sorted_nodes = toposort(out)
        grads = {n.name: None for n in sorted_nodes}
        grads["o"] = 1.0

        print(f"Initial grads: {grads}")

        for node in sorted_nodes:
            print(f"processing {node.name}")

            if not node.parents:
                continue

            parents = node.parents
            print(f"node {node.name} parents: {[p.name for p in parents]}")


            parents_grads = node.vjp(grads[node.name])
            # print(f"node {node.name} contributions to parent grads: {parents_grads}")

            for parent, parent_grad in zip(parents, parents_grads):
                if grads[parent.name]:
                    grads[parent.name] += parent_grad
                else:
                    grads[parent.name] = parent_grad

            print(f"grads after processing {node.name}: {grads}")
            # print(f"Node; {node}")
            # print(f"grads: {grads}")

        return grads

    x = Node(1.0, name="x")
    y = Node(2.0, name="y")

    out = f(x, y)

    nodes = toposort(out)

    grads = grad(f, at=(x, y))
    print(grads)
    # print(gradients)
