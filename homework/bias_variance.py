import numpy as np
import matplotlib.pyplot as plt

N = 10000
x = np.random.uniform(-1,1,N)
# reshape vector to (9,1) matrix
X = x.reshape((-1,1))
# g = 1.43*x
f = np.sin(np.pi * x)

def mse(x,y):
    return (x-y)**2

def bias(g,f):
    return (g-f)**2

def var(g_d, g_mean):
    return (g_d - g_mean)**2

def linear_regression(X,y):
    X_dagger = np.linalg.pinv(X)
    return np.dot(X_dagger, y)

w = []
for i in range(-1,N-1,1):
    points = np.array([[x[i]], [x[i+1]]])
    targets = np.array([[f[i]], [f[i+1]]])

    w.append(linear_regression(points, targets)[0])
w = np.array(w)
a_mean = np.mean(w)
g_mean = a_mean*x
print("a_mean: ", a_mean)

b = bias(g_mean,f)
print("bias: ", np.mean(b))

variance = []
hyp_set = np.dot(X, w.T)
for h in hyp_set.T:
    variance.append(np.mean(var(h,g_mean)))
variance = np.mean(variance)
print("var: ", variance)


plt.scatter(x,b)
plt.scatter(x,f)
plt.scatter(x,g_mean)
plt.show()


