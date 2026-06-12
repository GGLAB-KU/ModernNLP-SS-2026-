import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Toy Non-linear Data (XOR-like problem)
# Inputs (x1, x2), Targets (0 or 1)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# 2. Define a Simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Hidden layer: 2 inputs -> 4 hidden neurons
        self.hidden = nn.Linear(2, 4)
        self.activation = nn.Sigmoid()
        # Output layer: 4 hidden neurons -> 1 output prediction
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.activation(x) # Squashes output between 0 and 1
        return x

# Initialize components
model = SimpleNet()
criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.5) # Stochastic Gradient Descent with eta = 0.5

# 3. Training Loop with Visualization Caching
epochs = 2001
loss_history = []
epochs_to_plot = [0, 200, 500, 2000]
plot_data = {}

# Meshgrid for plotting the decision boundary surface later
x_span = np.linspace(-0.5, 1.5, 100)
y_span = np.linspace(-0.5, 1.5, 100)
xx, yy = np.meshgrid(x_span, y_span)
grid_data = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

for epoch in range(epochs):
    # Step A: Forward Pass
    predictions = model(X)

    # Step B: Compute Loss
    loss = criterion(predictions, Y)
    loss_history.append(loss.item())

    # Step C: Zero the Gradients (clear out old memory)
    optimizer.zero_grad()

    # Step D: Backward Pass (Backpropagation computes derivatives)
    loss.backward()

    # Step E: Gradient Descent Step (Updates weights using computed gradients)
    optimizer.step()

    # Cache intermediate state for visualization
    if epoch in epochs_to_plot:
        with torch.no_grad():
            grid_preds = model(grid_data).reshape(xx.shape).numpy()
            plot_data[epoch] = grid_preds

# 4. Plotting how the model learns over time
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

X_np = X.numpy()
Y_np = Y.squeeze().numpy()

for idx, epoch in enumerate(epochs_to_plot):
    ax = axes[idx]
    # Plot decision boundary surface
    contour = ax.contourf(xx, yy, plot_data[epoch], levels=50, cmap='coolwarm', alpha=0.6)

    # Plot original training points
    scatter = ax.scatter(X_np[:, 0], X_np[:, 1], c=Y_np, cmap='coolwarm', edgecolors='k', s=200, linewidth=2)

    ax.set_title(f"Epoch {epoch} | Loss: {loss_history[epoch]:.4f}", fontsize=12)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.grid(True, linestyle='--', alpha=0.5)

fig.colorbar(contour, ax=axes.ravel().tolist(), label='Model Prediction Probability')
plt.suptitle("Visualizing How the Model Learns a Non-Linear Space Through Epochs", fontsize=16, weight='bold')
plt.show()

# 5. Plot the Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='purple', linewidth=2)
plt.title("Loss Optimization Curve Across Training", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Binary Cross Entropy Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()