#include <iostream>
#include <map>
#include <set>
#include <type_traits>
#include <vector>
using namespace std;

// Unique id for each Node created
auto get_unique_id() -> int {
  static atomic<uint32_t> uid{0};
  return ++uid;
}

struct Node {

  int id;
  float val;
  vector<Node> parents;
  function<vector<float>(float, Node, Node)> grad_fn;

  // Constructor
  Node(float val, vector<Node> parents = {},
       function<vector<float>(float, Node, Node)> grad_fn = NULL)
      : id(get_unique_id()), val(val), parents(parents),
        grad_fn(grad_fn) {}

  // Equality checks id
  bool operator==(const Node &other) const {
    if (this->id == other.id) {
      return true;
    } else {
      return false;
    }
  }
  // Comparison always returns true
  bool operator<(const Node &other) const { return true; }
};

// Addition
auto add_grad(float g, Node x, Node y) -> vector<float> {
  return {g, g};
}

auto add(const Node x, const Node y) -> Node {
  return Node{
      x.val + y.val,
      {x, y},
      add_grad,
  };
}

// Subtraction
auto sub_grad(float g, Node x, Node y) -> vector<float> {
  return {g, -g};
}

auto sub(const Node x, const Node y) -> Node {
  return Node{
      x.val - y.val,
      {x, y},
      sub_grad,
  };
}

// Multiplication
auto mul_grad(float g, Node x, Node y) -> vector<float> {
  return {g * y.val, g * x.val};
}

auto mul(const Node x, const Node y) -> Node {
  return Node{
      x.val * y.val,
      {x, y},
      mul_grad,
  };
}

// Division
auto div_grad(float g, Node x, Node y) -> vector<float> {
  return {g / y.val, -(g * x.val) / (y.val * y.val)};
}

auto div(const Node x, const Node y) -> Node {
  return Node{
      x.val * y.val,
      {x, y},
      div_grad,
  };
}

auto test_function(float x, float y) -> Node {
  Node out = add(Node{x}, Node{y});
  return out;
}

auto dfs(const Node &node, const set<Node> &visited,
         vector<Node> &sorted_ndoes) -> void {
  const bool is_in = visited.find(node) != visited.end();
  cout << is_in;
}

auto toposort(Node node) -> vector<Node> {

  set<Node> visited;
  vector<Node> sorted_nodes;

  return {Node{1.0}};
}; // todo: will need to create a hash function for Node? or uuid

typedef auto float_func(function<vector<float>(vector<float>)>)
    -> function<vector<float>(vector<float>)>;
template <typename T>
auto grad(float_func f, vector<T> inputs)
    -> vector<float> { // todo switch int with function type

  // todo: tree_map inputs
  Node out = f(inputs);

  // to hold gradients of all nodes
  map<Node, float> grads;
  grads[out] = 1.0;

  // todo: toposort

  // todo: loop through toposort and accumulate gradients
  vector<float> _out = {grads[out]};
  return _out;
}

int main() {

  // // // for (int i = 0; i < 5; ++i) {
  // //   cout << get_unique_id() << ' ';
  // }
  Node first_node = Node{1.0};
  cout << first_node.id << '\n';
  Node out = test_function(1.0, 2.0);
  vector<float> grads =
      out.grad_fn(2.0, out.parents[0], out.parents[1]);
  for (auto i : grads) {
    cout << i << ' ';
  }

  set<Node> visited;
  visited.insert(out);



  auto visited_el = visited.begin();
  Node other = *visited_el;
  if (other == out){
    cout << "found!";
  }

  
  auto iter = visited.find(out);
  
  if (iter != visited.end()) {
    cout << "found!";
  }

  const bool is_in = visited.find(out) != visited.end();
  if (is_in) {
    cout << "blah";
  };
  return 0;
};
