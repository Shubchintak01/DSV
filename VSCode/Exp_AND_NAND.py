import numpy as np
from sklearn.neural_network import MLPClassifier

# Input data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# AND gate output
Y_AND = np.array([0, 0, 0, 1])

# NAND gate output
Y_NAND = np.array([1, 1, 1, 0])

# ------------------------------------------------
# AND Gate using Feed Forward Neural Network
# ------------------------------------------------

and_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='lbfgs',
    random_state=1,
    max_iter=1000
)

and_model.fit(X, Y_AND)

and_prediction = and_model.predict(X)

print("AND Gate")
print("A B | Output")

for i in range(len(X)):
    print(X[i][0], X[i][1], "|", and_prediction[i])

# ------------------------------------------------
# NAND Gate using Feed Forward Neural Network
# ------------------------------------------------

nand_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='lbfgs',
    random_state=1,
    max_iter=1000
)

nand_model.fit(X, Y_NAND)

nand_prediction = nand_model.predict(X)

print("\nNAND Gate")
print("A B | Output")
for i in range(len(X)):
    print(X[i][0], X[i][1], "|", nand_prediction[i])

from sklearn.metrics import accuracy_score

# Accuracy of AND gate
and_accuracy = accuracy_score(Y_AND, and_prediction)

# Accuracy of NAND gate
nand_accuracy = accuracy_score(Y_NAND, nand_prediction)

print("\nAND Gate Accuracy:", and_accuracy * 100, "%")
print("NAND Gate Accuracy:", nand_accuracy * 100, "%")