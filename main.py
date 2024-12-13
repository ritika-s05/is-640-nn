from engine import Value
from nn import MLP

# Input data and expected outputs
xs = [[2.0, 3.0, -1.0], 
      [3.0, -1.0, 0.5], 
      [0.5, 1.0, 1.0], 
      [1.0, 1.0, -1.0]]  

ys = [1.0, -1.0, -1.0, 1.0]  # Labels

# Initialize the neural network with 3 input, 2 hidden (4 neurons each), and 1 output layer
n = MLP(3, [4, 4, 1])

# Training loop
for k in range(20):
    # Forward pass: generate predictions
    ypred = [n(x) for x in xs]
    # Calculate loss as mean squared error
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Zero gradients, backpropagate, and update parameters
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # Gradient descent step
    for p in n.parameters():
        p.data += -0.1 * p.grad 

    print(k, loss.data)  # Output epoch and lossimport random
from engine import Value
from nn import MLP

# Create a simple dataset (e.g., XOR problem)
data = [
    ([2.0, 3.0], 1.0),
    ([3.0, -1.0], -1.0),
    ([1.0, 1.0], 1.0),
    ([2.0, -2.0], -1.0)
]

# Initialize a simple MLP model: 2 inputs, one hidden layer with 4 neurons, 1 output
model = MLP(2, [4, 1])

# Training loop
epochs = 20  # Number of iterations
learning_rate = 0.01

for k in range(epochs):
    # Forward pass: predict the output for each data point
    total_loss = Value(0)
    for x, y in data:
        x = [Value(xi) for xi in x]  # Convert inputs to Value objects
        y_pred = model(x)  # Forward pass
        loss = (y_pred - Value(y)) ** 2  # Mean squared error loss
        total_loss += loss
    
    # Backward pass: reset gradients, calculate new gradients
    model.zero_grad()  # Clear previous gradients
    total_loss.backward()  # Backpropagation
    
    # Update model parameters using gradient descent
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    # Print the progress: epoch and current loss
    print(k, total_loss.data)
