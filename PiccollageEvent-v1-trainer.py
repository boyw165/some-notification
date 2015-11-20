# coding=utf-8
"""
Version: 1.0
Description: Use MSE and linear regression to estimate when is most likely the
             user will open the application.
"""
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ex = np.loadtxt('./data_examples/usage-of-piccollage-weekly-per-user-week1.csv',
                delimiter=',',
                skiprows=1,
                usecols=(1, 2))
x_data = np.insert(ex[:, :-1], 0, 1, axis=1)
y_data = ex[:, ex.shape[1] - 1:]

# Construct the model.
# w_data = np.random.rand(1, 3)
# w_data = np.ones((1, 3), dtype=np.int)
w_data = np.array([[0, 0]])

print 'x_data=%s' % x_data
print '----------'
print 'y_data=%s' % y_data
print '----------'
print 'Initial w_data=%s' % w_data
print '----------'


# Is the cost function convex?

def hFn(x, w):
    """
    The hypothesis is:

        h(x) = w0 * x0 + w1 * (sin(x2) / x2), where x0 = 1.

    Note: sin(x)/x is called "cardinal sine function" or "sinc function".

    :param x: One dimension numpy.ndarray.
    :param w: One dimension numpy.ndarray.
    :return: Numerical number.
    """
    if not isinstance(x, np.ndarray) or \
        not isinstance(w, np.ndarray):
        raise TypeError('Should be type of numpy.ndarray!')
    if len(x) != 2 or len(w) != 2:
        raise ArithmeticError('The dimension does not match!')
    ret = w[0] * x[0] + w[1] * np.sinc(x[1])
    return ret


def costfunction(x, y, w):
    """
    Calculate the mean squared error. The formular is:

        J(w) = 1/m * sum(math.pow(h(x) - y))

    :param x: The x data.
    :param y: The y data.
    :param w: The weight.
    :return:
    """
    if not isinstance(x, np.ndarray) or \
        not isinstance(y, np.ndarray) or \
        not isinstance(w, np.ndarray):
        raise TypeError('Should be type of numpy.ndarray!')
    if x.shape[0] != y.shape[0]:
        raise ArithmeticError('The dimension does not match!')

    sum = 0
    amount = len(y)

    for m in range(amount):
        sum += math.pow(hFn(x[m], w[0]) - y[m], 2)

    sum /= (2 * amount)
    return sum


def gradientDescent(x, y, w):
    """
    Do gradient descent for once in terms of the given x, y and w.
    :param x:
    :param y:
    :param w:
    :return:
    """
    alpha = 0.5
    amount = len(y)
    w1 = w[0, 1]
    w2 = w[0, 2]
    der_w = w[:]

    # Get ∇J(w)
    for m in range(amount):
        err = hFn(x[m], w[0]) - y[m]
        x0 = x[m, 0]
        x1 = x[m, 1]
        # WORKAROUND: make them extremely small to 0.
        der_w[0, 0] = err * x0
        der_w[0, 1] = err * np.sin(w2 * x1) / x1
        der_w[0, 2] = err * w1 * np.sin(w2 * x1)
    der_w /= amount

    # J(w) := J(w) - alpha * ∇J(w)
    # for i in range(3):
    #     new_w[0, i] = w[0, i] - alpha * der_w[0, i]
    new_w = w - alpha * der_w

    return new_w


# Start the iterations.
print '> Before any training, the cost is %f' % costfunction(x_data, y_data,
                                                             w_data)
for i in range(10):
    new_w_data = gradientDescent(x_data, y_data, w_data)
    print '#%d from %s to %s' % (i, w_data, new_w_data)
    w_data = new_w_data
    # Break the loop for specific condition either like already converge or
    # diverge.
print '> After any training, the cost is %f' % costfunction(x_data, y_data,
                                                            w_data)

# Plot the result.
# plt.plot(x_data[:, 1], y_data[:, 0], 'r+')
# plt.show()
