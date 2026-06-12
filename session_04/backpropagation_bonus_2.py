import numpy as np

###############################################################################
#                       1) Analytic Graph Class                               #
###############################################################################
class AnalyticGraph:
    def __init__(self):
        # node_dict: { node_name : { 'forward_func':..., 'backward_func':..., 'parents':[...] } }
        self.node_dict = {}
        self.topo_order = []

    def add_node(self, name, forward_func, backward_func, parents=[]):
        """
        name: unique identifier for this node
        forward_func(*parents_vals) -> float
        backward_func(d_out, *parents_vals) -> { parentName: partial }
        parents: list of node names (strings)
        """
        self.node_dict[name] = {
            'forward_func': forward_func,
            'backward_func': backward_func,
            'parents': parents
        }

    def topological_sort(self):
        visited = set()
        order = []

        def dfs(node):
            if node not in visited:
                visited.add(node)
                for p in self.node_dict[node]['parents']:
                    dfs(p)
                order.append(node)

        for n in self.node_dict:
            dfs(n)
        return order

    def forward(self, inputs):
        """
        inputs: { nodeName -> value } for the input/parameter nodes
        returns: { nodeName -> computedValue } for all nodes
        """
        values = dict(inputs)
        if not self.topo_order:
            self.topo_order = self.topological_sort()

        for node in self.topo_order:
            if node not in values:  # must compute
                info = self.node_dict[node]
                f = info['forward_func']
                parent_vals = [values[p] for p in info['parents']]
                values[node] = f(*parent_vals)
        return values

    def backward(self, values, loss_node):
        """
        values: forward pass results
        loss_node: final node name (e.g. 'loss')
        returns: { nodeName -> d(Loss)/d(nodeValue) }
        """
        grads = {n: 0.0 for n in self.node_dict}
        grads[loss_node] = 1.0  # dLoss/dLoss = 1

        for node in reversed(self.topo_order):
            d_out = grads[node]  # d(Loss)/d(nodeValue)
            info = self.node_dict[node]
            if not info['parents']:
                continue
            bfunc = info['backward_func']
            parent_vals = [values[p] for p in info['parents']]
            local_partials = bfunc(d_out, *parent_vals)
            # local_partials = { parentName: partial w.r.t that parent }
            for p, dp in local_partials.items():
                grads[p] += dp
        return grads

###############################################################################
#                     2) Build Forward/Backward for Each Node                 #
###############################################################################

# For parameter or input nodes with no parents.
def param_forward():
    return None
def param_backward(d_out):
    return {}

def input_forward():
    return None
def input_backward(d_out):
    return {}

########################################################################
# Simple Example From the Book                                         #
########################################################################

a = 3
b = 1
c = -2

# Build graph anew
graph = AnalyticGraph()

# 1) Input nodes
graph.add_node("a", input_forward, input_backward, parents=[])
graph.add_node("b", input_forward, input_backward, parents=[])
graph.add_node("c", input_forward, input_backward, parents=[])

# 2) d = 2*b
graph.add_node(
    name="d",
    forward_func=lambda b: 2 * b,
    backward_func=lambda d_out, b: {
        "b": d_out * 2
    },
    parents=["b"]
)

# 3) e = a+d
graph.add_node(
    name="e",
    forward_func=lambda a, d: a + d,
    backward_func=lambda d_out, a, d: {
        "a": d_out,
        "d": d_out
    },
    parents=["a", "d"]
)

# 4) L = c*e
graph.add_node(
    name="loss",
    forward_func=lambda c, e: c * e,
    backward_func=lambda d_out, c, e: {
        "c": d_out * e,
        "e": d_out * c
    },
    parents=["c", "e"]
)

# Forward pass
inputs = {"a": a, "b": b, "c": c}
node_values = graph.forward(inputs)
loss_val = node_values["loss"]

print("Initial loss is: ", loss_val)

# Backward pass
grads = graph.backward(node_values, "loss")

print(grads)

print("=="*30)
print("=="*30)
print("=="*30)

###############################################################################
#                   3) Build & Train a 2-layer XOR network                    #
###############################################################################

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0, 1, 1, 0])

np.random.seed(42)

# 2-layer parameters
W1_11 = np.random.randn()*0.5
W1_12 = np.random.randn()*0.5
W1_21 = np.random.randn()*0.5
W1_22 = np.random.randn()*0.5
b1_1  = 0.0
b1_2  = 0.0

W2_1  = np.random.randn()*0.5
W2_2  = np.random.randn()*0.5
b2    = 0.0

lr = 0.5
epochs = 10000

for epoch in range(epochs):
    total_loss = 0.0

    for i in range(len(X)):
        x1_val, x2_val = X[i]
        y_val = Y[i]

        # Build graph anew
        graph = AnalyticGraph()

        # 1) Input nodes
        graph.add_node("x1", input_forward, input_backward, parents=[])
        graph.add_node("x2", input_forward, input_backward, parents=[])
        graph.add_node("y",  input_forward, input_backward, parents=[])

        # 2) Parameter nodes (no parents)
        graph.add_node("w1_11", param_forward, param_backward, parents=[])
        graph.add_node("w1_12", param_forward, param_backward, parents=[])
        graph.add_node("w1_21", param_forward, param_backward, parents=[])
        graph.add_node("w1_22", param_forward, param_backward, parents=[])
        graph.add_node("b1_1",  param_forward, param_backward, parents=[])
        graph.add_node("b1_2",  param_forward, param_backward, parents=[])

        graph.add_node("w2_1",  param_forward, param_backward, parents=[])
        graph.add_node("w2_2",  param_forward, param_backward, parents=[])
        graph.add_node("b2",    param_forward, param_backward, parents=[])

        # 3) z1 = x1*w1_11 + x2*w1_21 + b1_1
        graph.add_node(
            name="z1",
            forward_func=lambda x1,x2,w1_11,w1_21,b1_1: x1*w1_11 + x2*w1_21 + b1_1,
            backward_func=lambda d_out, x1,x2,w1_11,w1_21,b1_1: {
                "x1":    d_out*w1_11,
                "w1_11": d_out*x1,
                "x2":    d_out*w1_21,
                "w1_21": d_out*x2,
                "b1_1":  d_out
            },
            parents=["x1","x2","w1_11","w1_21","b1_1"]
        )

        # z2 = x1*w1_12 + x2*w1_22 + b1_2
        graph.add_node(
            name="z2",
            forward_func=lambda x1,x2,w1_12,w1_22,b1_2: x1*w1_12 + x2*w1_22 + b1_2,
            backward_func=lambda d_out, x1,x2,w1_12,w1_22,b1_2: {
                "x1":    d_out*w1_12,
                "w1_12": d_out*x1,
                "x2":    d_out*w1_22,
                "w1_22": d_out*x2,
                "b1_2":  d_out
            },
            parents=["x1","x2","w1_12","w1_22","b1_2"]
        )

        # a1 = sigmoid(z1)
        graph.add_node(
            name="a1",
            forward_func=lambda z1: 1.0/(1.0 + np.exp(-z1)),
            backward_func=lambda d_out, z1: {
                "z1": d_out * (
                    (1.0/(1.0 + np.exp(-z1))) *
                    (1 - 1.0/(1.0 + np.exp(-z1)))
                )
            },
            parents=["z1"]
        )

        # a2 = sigmoid(z2)
        graph.add_node(
            name="a2",
            forward_func=lambda z2: 1.0/(1.0 + np.exp(-z2)),
            backward_func=lambda d_out, z2: {
                "z2": d_out * (
                    (1.0/(1.0 + np.exp(-z2))) *
                    (1 - 1.0/(1.0 + np.exp(-z2)))
                )
            },
            parents=["z2"]
        )

        # z3 = a1*w2_1 + a2*w2_2 + b2
        graph.add_node(
            name="z3",
            forward_func=lambda a1,a2,w2_1,w2_2,b2: a1*w2_1 + a2*w2_2 + b2,
            backward_func=lambda d_out, a1,a2,w2_1,w2_2,b2: {
                "a1":   d_out*w2_1,
                "w2_1": d_out*a1,
                "a2":   d_out*w2_2,
                "w2_2": d_out*a2,
                "b2":   d_out
            },
            parents=["a1","a2","w2_1","w2_2","b2"]
        )

        # a3 = sigmoid(z3)
        graph.add_node(
            name="a3",
            forward_func=lambda z3: 1.0/(1.0 + np.exp(-z3)),
            backward_func=lambda d_out, z3: {
                "z3": d_out * (
                    (1.0/(1.0 + np.exp(-z3))) *
                    (1 - 1.0/(1.0 + np.exp(-z3)))
                )
            },
            parents=["z3"]
        )

        # loss = 0.5*(a3 - y)^2
        graph.add_node(
            name="loss",
            forward_func=lambda a3, y: 0.5*(a3 - y)**2,
            backward_func=lambda d_out, a3, y: {
                "a3": d_out*(a3 - y),
                "y":  d_out*(-(a3 - y))
            },
            parents=["a3","y"]
        )

        # Forward pass
        inputs = {
            "x1": x1_val,
            "x2": x2_val,
            "y":  y_val,
            "w1_11": W1_11, "w1_12": W1_12, "w1_21": W1_21, "w1_22": W1_22,
            "b1_1": b1_1,   "b1_2": b1_2,
            "w2_1": W2_1,   "w2_2": W2_2,
            "b2":   b2
        }
        values = graph.forward(inputs)
        sample_loss = values["loss"]
        total_loss += sample_loss

        # Backward pass
        grads = graph.backward(values, "loss")

        # Update parameters
        W1_11 -= lr * grads["w1_11"]
        W1_12 -= lr * grads["w1_12"]
        W1_21 -= lr * grads["w1_21"]
        W1_22 -= lr * grads["w1_22"]
        b1_1  -= lr * grads["b1_1"]
        b1_2  -= lr * grads["b1_2"]

        W2_1  -= lr * grads["w2_1"]
        W2_2  -= lr * grads["w2_2"]
        b2    -= lr * grads["b2"]

    # Print average loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, avg loss={total_loss/len(X):.6f}")

print("\nFinal predictions:")
for i in range(len(X)):
    x1, x2 = X[i]
    z1 = x1*W1_11 + x2*W1_21 + b1_1
    z2 = x1*W1_12 + x2*W1_22 + b1_2
    a1 = 1.0/(1.0 + np.exp(-z1))
    a2 = 1.0/(1.0 + np.exp(-z2))
    z3 = a1*W2_1 + a2*W2_2 + b2
    a3 = 1.0/(1.0 + np.exp(-z3))
    print(f"Input={X[i]}, pred={a3:.4f}, target={Y[i]}")
