import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Input data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# XOR output
Y_XOR = np.array([0, 1, 1, 0])

# -----------------------------------------
# XOR using Feed-Forward Neural Network
# -----------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='lbfgs',
    random_state=1,
    max_iter=1000
)

# Train the neural network
model.fit(X, Y_XOR)

# Predict output
prediction = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(Y_XOR, prediction)

# Display results
print("EX-OR (XOR) Gate")
print("A B | Actual | Predicted")

for i in range(len(X)):
    print(
        X[i][0],
        X[i][1],
        " | ",
        Y_XOR[i],
        "     | ",
        prediction[i]
    )

print("\nXOR Gate Accuracy:",
      accuracy * 100, "%")

