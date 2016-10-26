import numpy as np
import matplotlib.pyplot as plt
import random

def target(x):
    """ target function for y """
    return np.sin(np.pi * x)

def bias(g,f):
    """ calculate bias """
    return (g-f)**2

def var(g_d, g_mean):
    """ calculate variance """
    return (g_d - g_mean)**2

def linear_regression(X,y):
    """ do one linear regression step """
    X_dagger = np.linalg.pinv(X)
    return np.dot(X_dagger, y)

def generate_data(N, h):
    """ generate x vector, X matrix, y vector, and empty w vector """
    x = np.array([np.linspace(-1,1,N)]).T
    # x = np.array([np.random.uniform(-1,1,N)]).T
    y = target(x)
    X = h(x,N)
    w = np.ndarray(shape = (X.shape[1],0))
    return x,X,y,w

if __name__ == "__main__":
    N = 100
    # g = 1.43*x

    # h(x) = b
    # x,X,y,w = generate_data(N, lambda x,N: np.column_stack([np.ones(N)]))

    # h(x) = a*x
    x,X,y,w = generate_data(N, lambda x,N: x.reshape((-1,1)) )

    # h(x) = a*x + b
    # x,X,y,w = generate_data(N, lambda x,N: np.column_stack([np.ones(N), x]))

    # h(x) = a*x^2
    # x,X,y,w = generate_data(N, lambda x,N: np.column_stack([x**2]))

    # h(x) = a*x^2 + b
    # x,X,y,w = generate_data(N, lambda x,N: np.column_stack([np.ones(N), x**2]))

    f = np.sin(np.pi * x)

    for i in range(N):
        rnd_idx = np.random.randint(N,size=2)
        x_points = X[rnd_idx]
        y_points = y[rnd_idx]
        # get last column of target
        w = np.hstack([w, linear_regression(x_points, y_points)])

    # mean of each column
    a_mean = np.mean(w, axis=1,keepdims=True)
    # g_mean = a_mean*x
    g_mean = np.dot(X,a_mean)
    print("a_mean: ", a_mean)
    print(a_mean.shape, w.shape, g_mean.shape)

    b = bias(g_mean,f)
    print("bias: ", np.mean(b))

    variance = []
    hyp_set = np.dot(X, w)
    for h in hyp_set.T:
        variance.append(np.mean(var(h,g_mean.T)))
    variance = np.mean(variance)
    print("var: ", variance)

    # do the plots
    for h in hyp_set.T:
        plt.plot(x,h,color='grey', alpha=0.1)
    plt.plot(x,b,color='pink',label="bias")
    plt.scatter(x,f,color='red',label="target")
    plt.scatter(x,g_mean,label="g_mean")
    plt.legend()
    plt.grid()
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.show()


