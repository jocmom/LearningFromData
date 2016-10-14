import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt


def target(feature):
    slope = (point2[1]- point1[1]) / (point2[0] - point1[0])
    if feature[1] > (slope * (feature[0] - point1[0]) + point1[1]):
        return 1
    return -1

def generate_data():
    return [[1., np.random.uniform(space[0], space[1]), np.random.uniform(space[0], space[1])] for i in range(N)]

def generate_output(features):
    return [target(f[1:3]) for f in features]

def do_plot(features, y, weights):
    plt.xlim(space[0], space[1])
    plt.ylim(space[0], space[1])
    # calculate the linear function from the PLA weights
    x1 = np.linspace(space[0], space[1], num=20)
    x2 = -1./weights[2] * (weights[1] * x1 + weights[0])
    # plot points above and below line in different colors
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(features[i][1], features[i][2])
        else:
            plt.scatter(features[i][1], features[i][2], color='g')

    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='r')
    plt.plot(x1,x2,color='pink')
    plt.show()

def linear_regression(X,y):
    X_dagger = pinv(X)
    return np.dot(X_dagger, y)

if __name__ == "__main__":
    d = 2
    N = 1000
    space = (-1,1)
    runs = 1000
    E_in = 0
    avg_iterations = 0.

    for i in range(runs):

        point1 = (np.random.uniform(-1,1), np.random.uniform(-1,1))
        point2 = (np.random.uniform(-1,1), np.random.uniform(-1,1))

        features = generate_data()
        features_test = generate_data()
        y = generate_output(features)
        y_test = generate_output(features_test)
        weights = linear_regression(np.array(features), np.array(y))

        h = np.sign(np.dot(features, weights))
        for i in range(len(h)):
            if h[i] != y[i]:
                E_in += 1

    print("E_in", (E_in/runs)/N)

    do_plot(features, y, weights)
