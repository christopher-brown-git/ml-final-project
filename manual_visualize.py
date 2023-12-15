import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure(figsize=(10, 6))

tests = ["Training \n[8, 4, 1]", "Training \n[20, 10, 1]", "Training \n[40, 20, 1]", "Training \n[100, 50, 1]", "Training \n[4, 5, 2, 7, 3, 2, 1]"]

accuracies = [50.36, 50.36, 50.36, 49.64, 49.59]
plt.bar(tests, accuracies, color="blue", width=0.4)

plt.xlabel("Different Models Created")
plt.ylabel("Accuracy")
plt.title("Comparing different neural network architectures with dropout=0.0 and learning_rate=0.5")
plt.ylim(min(accuracies)-.10, max(accuracies)+.05)

plt.show()