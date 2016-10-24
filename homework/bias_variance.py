import numpy as np
import matplotlib.pyplot as plt
import random

def target(x):
    return np.sin(np.pi * x)

def bias(g,f):
    return (g-f)**2

def var(g_d, g_mean):
    return (g_d - g_mean)**2

def linear_regression(X,y):
    X_dagger = np.linalg.pinv(X)
    return np.dot(X_dagger, y)

if __name__ == "__main__":
    N = 1000
    # x = np.random.uniform(-1,1,N)
    x = np.linspace(-1,1,N)
    # reshape vector to (N,1) matrix
    X = x.reshape((-1,1))
    transform = lambda x,n: np.random.choice(x,n).reshape((-1,1))
    w = np.ndarray(shape = (1, 0))
    X = np.vstack([np.ones(N), x]).T
    transform = lambda x,n: np.vstack([np.ones(n), np.random.choice(x,n)]).T
    w = np.ndarray(shape = (2, 0))
    # g = 1.43*x
    f = np.sin(np.pi * x)

    for i in range(N):
        # points = np.array([[x[i]], [x[i+1]]])
        points = transform(x, 2)
        # point1 = np.ones(2).reshape((-1,1))
        # point2 = np.random.choice(x, 2).reshape((-1,1))
        # points = np.hstack([point1, point2])
        """ get last column of target """
        # w.append(linear_regression(points, target(points[:,[-1]])))
        # print(w)
        w = np.hstack([w, linear_regression(points, target(points[:,[-1]]))])

    # print(w)
    # w = np.array(w)
    # print(w.T)
    a_mean = np.mean(w)
    g_mean = a_mean*x
    print("a_mean: ", a_mean)

    b = bias(g_mean,f)
    print("bias: ", np.mean(b))

    variance = []
    hyp_set = np.dot(X, w)
    for h in hyp_set.T:
        variance.append(np.mean(var(h,g_mean)))
    variance = np.mean(variance)
    print("var: ", variance)


    for h in hyp_set.T:
        plt.plot(x,h,color='grey', alpha=0.1)
    plt.plot(x,b,color='pink',label="bias")
    plt.scatter(x,f,color='red',label="target")
    plt.scatter(x,g_mean,label="g_mean")
    plt.legend()
    plt.show()


