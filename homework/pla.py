import numpy as np
import random
import matplotlib.pyplot as plt


def target(feature):
    slope = (point2[1]- point1[1]) / (point2[0] - point1[0])
    if feature[1] > (slope * (feature[0] - point1[0]) + point1[1]):
        return 1
    return -1

def generate_data():
    return [[1., random.uniform(space[0], space[1]), random.uniform(space[0], space[1])] for i in range(N)]

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

def pla(features, y, weights):
    iterations = 0

    while True:
        idx_misclassified = []
        # get hypothesis values
        h = np.sign(np.dot(features, weights))

        for i in range(len(h)):
            if h[i] != y[i]:
                idx_misclassified.append(i)
        if not idx_misclassified:
            break

        n = random.choice(idx_misclassified)
        weights += features[n].T * y[n]# * features[n]

        iterations += 1

    return iterations, weights

def test(features_test, y_test, weights):
    err = 0.
    h = np.sign(np.dot(features_test, weights))
    for i in range(len(y_test)):
        if y_test[i] != h[i]:
            err +=1

    return err/len(y_test)



if __name__ == "__main__":
    d = 2
    N = 200
    space = (-1,1)
    runs = 1000
    avg_iterations = 0.
    avg_err = 0.

    for i in range(runs):

        point1 = (np.random.uniform(-1,1), np.random.uniform(-1,1))
        point2 = (np.random.uniform(-1,1), np.random.uniform(-1,1))

        features = generate_data()
        features_test = generate_data()
        y = generate_output(features)
        y_test = generate_output(features_test)
        weights = [0. for i in range(3)]

        iterations, weights = pla(np.array(features), np.array(y), np.array(weights))
        avg_err += test(features_test, y_test, weights)
        avg_iterations += iterations

    print("Average Iterations for N=%i in %i runs: %f" % (N, runs, avg_iterations/runs))
    print("Average Error rate for N=%i in %i runs: %f" % (N, runs, avg_err/runs))

    do_plot(features, y, weights)


