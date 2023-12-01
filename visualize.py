import matplotlib.pyplot as plt
import numpy as np
import os

path = "10krows.a"
#return 0 for error and 1 otherwise
def visualize():
    if not os.path.isfile(path):
        return 0
    
    title = ""
    info = {}

    file = open(path, "r")

    for line in file:
        if line[0:2] == "**":
            if len(info) > 0:
                #plot
                tests = list(info.keys())
                accuracies = [float(x[1:-1]) for x in info.values()]

                fig = plt.figure(figsize=(10, 6))

                plt.bar(tests, accuracies, color="blue", width=0.4)

                plt.xlabel("Different Models Created")
                plt.ylabel("Accuracy")
                plt.title(title)
                plt.ylim(min(accuracies), max(accuracies))

                plt.show()

            title = line[2:-3]
            info = {}
        elif line != '\n':
            arr = line.split(":")
            k = arr[0]
            v = arr[1][1:-1]

            info[k] = v
    file.close()

    return 1

def main():
    visualize()

if __name__ == "__main__":
    main()