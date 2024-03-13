from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(int(datetime.now().timestamp()))

inputs = np.random.uniform(-8, 8, size=(20, 2))
weights = np.array([[1, 5], [6, 8]]) # матриця ваг
b1 = -5 # зміщення

def neuron(inputs, weights, b): # Вихід нейрона на основі вхідних даних
    s = np.dot(weights, inputs)
    s = sum(s)+b

    if s > 0:
        return 1
    elif s < 0:
        return -1
    else:
        return 0

def plot_perceptron(inputs, weights, b, resolution=1000): # графік
    x1_vals = np.linspace(-8, 8, resolution)
    x2_vals = np.linspace(-8, 8, resolution)

    xx, yy = np.meshgrid(x1_vals, x2_vals)
    zz = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            zz[i, j] = neuron(np.array([[xx[i, j], yy[i, j]]]).transpose(), weights, b)

    plt.contourf(xx, yy, zz, levels=[-1, 0, 1], colors=['pink', 'purple'], alpha=0.5)

    for input_point in inputs:
        output = neuron(input_point, weights, b)
        color = 'white' if output > 0 else 'brown'
        plt.plot(input_point[0], input_point[1], marker='o', markersize=10, color=color)

    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Perceptron Decision Boundary with Inputs')
    plt.colorbar()
    plt.show()

print(f"equation of the line of the neuron is:{weights[0][0]+weights[0][1]}p1+{weights[1][0]+weights[1][1]}p2+{b1}")

plot_perceptron(inputs, weights, b1)
