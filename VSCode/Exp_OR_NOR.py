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

# OR gate output
Y_OR = np.array([0, 1, 1, 1])

# NOR gate output
Y_NOR = np.array([1, 0, 0, 0])

# -----------------------------------------
# OR Gate using Feed-Forward Neural Network
# -----------------------------------------

or_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='lbfgs',
    random_state=1,
    max_iter=1000
)

# Train the model
or_model.fit(X, Y_OR)

# Prediction
or_prediction = or_model.predict(X)

# Accuracy
or_accuracy = accuracy_score(Y_OR, or_prediction)

print("OR Gate")
print("A B | Actual | Predicted")

for i in range(len(X)):
    print(X[i][0], X[i][1], " | ", Y_OR[i],
          "     | ", or_prediction[i])

print("OR Gate Accuracy:", or_accuracy * 100, "%")

# -----------------------------------------
# NOR Gate using Feed-Forward Neural Network
# -----------------------------------------

nor_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='lbfgs',
    random_state=1,
    max_iter=1000
)

# Train the model
nor_model.fit(X, Y_NOR)

# Prediction
nor_prediction = nor_model.predict(X)

# Accuracy
nor_accuracy = accuracy_score(Y_NOR, nor_prediction)

print("\nNOR Gate")
print("A B | Actual | Predicted")

for i in range(len(X)):
    print(X[i][0], X[i][1], " | ", Y_NOR[i],
          "     | ", nor_prediction[i])

print("NOR Gate Accuracy:", nor_accuracy * 100, "%")