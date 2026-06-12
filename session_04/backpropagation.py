import time

# --- THE SETUP ---
# Input (x), Current Weight Knob (w), Target Goal (y_target)
x = 2.0
w = 3.0
y_target = 10.0

print("=== 1. FORWARD PASS (Making the Skateboard) ===")
# Step A: The multiplication node
z = w * x
print(f"Node Z (w * x)       ->  3.0 * 2.0 = {z}")

# Step B: The prediction (let's say our model outputs z directly)
y_pred = z
print(f"Prediction (y_pred)  ->  {y_pred}")

# Step C: The Judge (Loss Function: Mean Squared Error style)
# Loss = 0.5 * (y_pred - y_target)^2
loss = 0.5 * (y_pred - y_target) ** 2
print(f"Final Loss Matrix    ->  0.5 * ({y_pred} - {y_target})^2 = {loss}")
print("-" * 50)

time.sleep(1) # Dramatic pause for the TA session

print("=== 2. BACKWARD PASS (Passing the Blame) ===")

# Step A: How much does the Loss change if y_pred changes?
# Derivative of 0.5 * (y_pred - y_target)^2 with respect to y_pred is: (y_pred - y_target)
d_loss_by_d_ypred = y_pred - y_target
print(f"1. Upstream Blame from Judge (dLoss/dy_pred): {d_loss_by_d_ypred}")
print("   Intuition: 'Hey! Your prediction is short by 4 units. Make it bigger!'")

# Step B: How much does y_pred change if the weight knob (w) changes?
# Since y_pred = z = w * x, the derivative with respect to w is just x.
d_ypred_by_dw = x
print(f"2. Local Impact of the Weight Knob (dy_pred/dw): {d_ypred_by_dw}")
print(f"   Intuition: 'Because the input x is {x}, turning the knob up by 1 adds {x} to the output.'")

# Step C: The Chain Rule (Multiply them together)
# Total Blame = Upstream Blame * Local Impact
d_loss_by_dw = d_loss_by_d_ypred * d_ypred_by_dw
print(f"3. Combined Final Gradient (dLoss/dw): {d_loss_by_dw}")
print(f"   Calculation: {d_loss_by_d_ypred} * {d_ypred_by_dw} = {d_loss_by_dw}")
print("-" * 50)

time.sleep(1)

print("=== 3. GRADIENT DESCENT (Adjusting the Knob) ===")
learning_rate = 0.1
w_new = w - (learning_rate * d_loss_by_dw)

print(f"Old Weight Knob: {w}")
print(f"Adjustment:     - ({learning_rate} * {d_loss_by_dw})")
print(f"New Weight Knob: {w_new}")

# Let's quickly verify if the loss actually went down with the new weight!
z_new = w_new * x
loss_new = 0.5 * (z_new - y_target) ** 2
print(f"🎉 New Loss with updated weight: {loss_new} (Down from {loss}!)")