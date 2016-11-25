import numpy as np
import matplotlib.pyplot as plt


def linear_regression(X, y):
    """ do one linear regression step """
    X_dagger = np.linalg.pinv(X)
    return np.dot(X_dagger, y)


in_data = np.genfromtxt("in.dta")
out_data = np.genfromtxt("out.dta")
# get all features,
X_in = in_data[:, 0:-1]
X_out = out_data[:, 0:-1]
# get y, last column
y_in = in_data[:, -1]
y_in = np.array([y_in]).T
y_out = out_data[:, -1]
y_out = np.array([y_out]).T

transform_functions = [
    lambda x: np.ones(len(x)),
    lambda x: x[:, 0],
    lambda x: x[:, 1],
    lambda x: x[:, 0]**2,
    lambda x: x[:, 1]**2,
    lambda x: x[:, 0] * x[:, 1],
    # lambda x: np.abs(x[:, 0] - x[:, 1]),
    # lambda x: np.abs(x[:, 0] + x[:, 1])
]

# 1.
# X_train = X_in[0:-10, :]
# y_train = y_in[0:-10]
# X_test = X_in[-10:, :]
# y_test = y_in[-10:]
# 2.
X_train = X_in[0:-10, :]
y_train = y_in[0:-10]
X_test = X_out
y_test = y_out
# 3.
# X_train = X_in[-10:, :]
# y_train = y_in[-10:]
# X_test = X_in[0:-10, :]
# y_test = y_in[0:-10]
# 4.
# X_train = X_in[-10:, :]
# y_train = y_in[-10:]
# X_test = X_out
# y_test = y_out


# transformation of training and test data
Z = np.ndarray(shape = (X_train.shape[0],0))
Z_test = np.ndarray(shape = (X_test.shape[0],0))

# increasing z space and doing linear regression on it
for k, phi in enumerate(transform_functions):
    Z = np.column_stack([Z, phi(X_train)])
    Z_test = np.column_stack([Z_test, phi(X_test)])
    w = linear_regression(Z, y_train)
    h = np.sign(np.dot(w.T, Z_test.T))
    err_cnt = 0
    for i in range(len(y_test)):
        if y_test.flatten()[i] != h.flatten()[i] :
            err_cnt += 1
    print("There are %i classification errors for k=%i" % (err_cnt, k))
    print("Validation error: %f" % (err_cnt/len(y_test)))

x = np.linspace(-1,1).reshape((-1,1))
x = np.column_stack([x,x])


dim = np.linspace(-1,1,100)
xx, yy = np.meshgrid(dim, dim)
XX = np.array([xx.flatten(), yy.flatten()]).T
ZZ = np.ndarray(shape=(XX.shape[0], 0))
for k,phi in enumerate(transform_functions):
    ZZ = np.column_stack([ZZ, phi(XX)])
hh = np.dot(w.T, ZZ.T).reshape(100,100)
plt.contour(xx, yy, hh, levels=[0], colors=['green'])

# plot data
plt.scatter(X_in[:, 0], X_in[:, 1], c=y_in, s=200)
plt.scatter(X_out[:, 0], X_out[:, 1], c=y_out, s=50)
# plt.scatter(X_in[:, 0], transform_functions[5](X_in), c=y_in, s=200)
# plt.scatter(X_out[:, 0], transform_functions[7](X_out), c=y_out, s=50)
plt.show()
