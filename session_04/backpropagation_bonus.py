import numpy as np

class ComputationGraph:
    def __init__(self):
        self.nodes = []
        self.edges = {}

    def add_node(self, node_name, func=None, parents=[]):
        self.nodes.append(node_name)
        self.edges[node_name] = {'func': func, 'parents': parents}

    def forward(self, inputs):
        """
        Forward pass: given input node values, compute values for all nodes
        in topological order. Returns a dict {node_name: value}.
        """
        values = dict(inputs)
        for node in self.topological_sort():
            if node not in values:
                func = self.edges[node]['func']
                parents = self.edges[node]['parents']
                parent_vals = [values[p] for p in parents]
                values[node] = func(*parent_vals)
        return values

    def backward(self, loss_node, node_values):
        """
        Numeric backprop: For each node, compute dLoss/dNode using partial
        derivatives. node_values is the dictionary of forward-pass values.
        Returns {node_name -> gradient}.
        """
        grads = {n: 0.0 for n in self.nodes}
        grads[loss_node] = 1.0

        for node in reversed(self.topological_sort()):
            parents = self.edges[node]['parents']
            if not parents:
                continue
            # Compute partial derivatives of 'node' wrt each parent
            local_grads = self.numeric_grads(node, node_values)
            for p, dlocal in local_grads.items():
                grads[p] += grads[node] * dlocal

        return grads

    def numeric_grads(self, node_name, node_values, eps=1e-5):
        """
        Compute partial derivatives d(node_name)/d(parent) by reusing the
        parent's actual forward-pass value from node_values.
        """
        func = self.edges[node_name]['func']
        parents = self.edges[node_name]['parents']

        def local_forward(par_val_changes):
            """
            Recompute this node's output by substituting changed parent values
            into the 'func' call, while using stored forward-pass values for
            others.
            """
            parent_vals = []
            for p in parents:
                if p in par_val_changes:
                    parent_vals.append(par_val_changes[p])
                else:
                    parent_vals.append(node_values[p])
            return func(*parent_vals)

        grads_dict = {}
        for p in parents:
            orig_val = node_values[p]

            # Evaluate f( ... p + eps ...)
            f_plus = local_forward({p: orig_val + eps})
            # Evaluate f( ... p - eps ...)
            f_minus = local_forward({p: orig_val - eps})

            grads_dict[p] = (f_plus - f_minus) / (2.0 * eps)

        return grads_dict

    def topological_sort(self):
        visited = set()
        order = []

        def dfs(n):
            if n not in visited:
                visited.add(n)
                for p in self.edges[n]['parents']:
                    dfs(p)
                order.append(n)

        for node in self.nodes:
            dfs(node)
        return order

########################################################################
# 2-Layer XOR Network Using the Improved Numeric-Gradient Computation  #
########################################################################

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def squared_error(a, y):
    return 0.5 * (a - y)**2

# XOR Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Parameter initialization
np.random.seed(42)
W1_11 = np.random.randn() * 0.5
W1_12 = np.random.randn() * 0.5
W1_21 = np.random.randn() * 0.5
W1_22 = np.random.randn() * 0.5
b1_1  = 0.0
b1_2  = 0.0

W2_1  = np.random.randn() * 0.5
W2_2  = np.random.randn() * 0.5
b2    = 0.0

learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    total_loss = 0.0

    # Process each sample individually (online/batch=1 training)
    for i in range(len(X)):
        # Build computation graph
        graph = ComputationGraph()

        # Input nodes
        graph.add_node("x1")
        graph.add_node("x2")

        # Hidden layer
        graph.add_node("w1_11"); graph.add_node("w1_12")
        graph.add_node("w1_21"); graph.add_node("w1_22")
        graph.add_node("b1_1");  graph.add_node("b1_2")

        # z1 = x1*w1_11 + x2*w1_21 + b1_1
        graph.add_node(
            "z1",
            func=lambda x1, x2, w11, w21, b11: x1*w11 + x2*w21 + b11,
            parents=["x1", "x2", "w1_11", "w1_21", "b1_1"]
        )
        # z2 = x1*w1_12 + x2*w1_22 + b1_2
        graph.add_node(
            "z2",
            func=lambda x1, x2, w12, w22, b12: x1*w12 + x2*w22 + b12,
            parents=["x1", "x2", "w1_12", "w1_22", "b1_2"]
        )
        # a1 = sigmoid(z1), a2 = sigmoid(z2)
        graph.add_node("a1", func=sigmoid, parents=["z1"])
        graph.add_node("a2", func=sigmoid, parents=["z2"])

        # Output layer
        graph.add_node("w2_1")
        graph.add_node("w2_2")
        graph.add_node("b2")

        # z3 = a1*w2_1 + a2*w2_2 + b2
        graph.add_node(
            "z3",
            func=lambda a1, a2, w21, w22, b2: a1*w21 + a2*w22 + b2,
            parents=["a1", "a2", "w2_1", "w2_2", "b2"]
        )
        graph.add_node("a3", func=sigmoid, parents=["z3"])

        # Loss node
        graph.add_node(
            "loss",
            func=lambda a3: squared_error(a3, y[i]),
            parents=["a3"]
        )

        # Forward pass
        inputs = {
            "x1": X[i, 0],  "x2": X[i, 1],
            "w1_11": W1_11, "w1_12": W1_12,
            "w1_21": W1_21, "w1_22": W1_22,
            "b1_1": b1_1,   "b1_2": b1_2,
            "w2_1": W2_1,   "w2_2": W2_2,
            "b2": b2
        }
        node_values = graph.forward(inputs)
        sample_loss = node_values["loss"]
        total_loss += sample_loss

        # Backward pass
        grads = graph.backward("loss", node_values)

        # Update parameters
        W1_11 -= learning_rate * grads["w1_11"]
        W1_12 -= learning_rate * grads["w1_12"]
        W1_21 -= learning_rate * grads["w1_21"]
        W1_22 -= learning_rate * grads["w1_22"]
        b1_1  -= learning_rate * grads["b1_1"]
        b1_2  -= learning_rate * grads["b1_2"]

        W2_1  -= learning_rate * grads["w2_1"]
        W2_2  -= learning_rate * grads["w2_2"]
        b2    -= learning_rate * grads["b2"]

    # Print average loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(X):.6f}")

# Test the trained network
print("\nFinal Predictions:")
for i in range(len(X)):
    x1, x2 = X[i]
    z1 = x1*W1_11 + x2*W1_21 + b1_1
    z2 = x1*W1_12 + x2*W1_22 + b1_2
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)

    z3 = a1*W2_1 + a2*W2_2 + b2
    a3 = sigmoid(z3)
    print(f"Input: {X[i]} -> Predicted: {a3:.4f} (Target: {y[i]})")