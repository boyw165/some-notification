# coding=utf-8
"""
Version: 1.0
Description: Use MSE and linear regression with hypothesis

                h(x) = w0*x0 + w1*sin(w2 + w3*x1), where x0=1, w3=0.25

             to estimate when is most likely the user will open the application.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

ex = np.loadtxt('./data_examples/usage-of-piccollage-weekly-per-user-week1.csv',
                delimiter=',',
                skiprows=1,
                usecols=(1, 2))
x_data = np.insert(ex[:, :-1], 0, 1, axis=1)
y_data = ex[:, ex.shape[1] - 1:]

# Construct the model. e.g. np.random.rand(...), np.ones((...), dtype=np.int)
w_data = np.array([[0, 60, 0, 1.0 / 4]])

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

        h(x) = w0 * x0 + w1 * sinc(w2 + w3 * x1)

    :param x: One dimension numpy.ndarray.
    :param w: One dimension numpy.ndarray.
    :return: Numerical number.
    """
    if not isinstance(x, np.ndarray) or \
        not isinstance(w, np.ndarray):
        raise TypeError('Should be type of numpy.ndarray!')
    ret = w[0] * x[0] + w[1] * np.sin(w[2] + w[3] * x[1])
    return ret


def costfunction(x, y, w):
    """
    Calculate the mean squared error. The formular is:

        J(w) = 1/(2 * m) * sum(math.pow(h(x) - y)), where m is the amount of
        given examples.

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
    :param x: The given x samples.
    :param y: The given y samples.
    :param w: The initial weights sample.
    :return: The renew weights sample.
    """
    alpha = 0.001
    amount = len(y)
    der_w = w.copy()
    w1 = w[0, 1]
    w2 = w[0, 2]
    w3 = w[0, 3]

    # Get ∇J(w)
    for m in range(amount):
        err = hFn(x[m], w[0]) - y[m]
        x0 = x[m, 0]
        x1 = x[m, 1]
        some_val = w2 + w3 * x1
        der_w[0, 0] = err * x0
        der_w[0, 1] = err * np.sin(some_val)
        der_w[0, 2] = err * np.cos(some_val) * w1
        # der_w[0, 3] = err * np.cos(some_val) * w1 * x1
    der_w /= amount

    # J(w) := J(w) - alpha * ∇J(w)
    new_w = w - alpha * der_w

    return new_w


# Start the iterations.
cost1 = costfunction(x_data, y_data, w_data)
for i in range(50000):
    new_w_data = gradientDescent(x_data, y_data, w_data)
    print '#%d weights from %s to %s' % (i, w_data, new_w_data)
    w_data = new_w_data
    # Break the loop for specific condition either like already converge or
    # diverge.
print '> Before training, the cost is %f' % cost1
print '> After training, the cost is %f' % costfunction(x_data, y_data, w_data)

# Plot the original training data.
plt.xlabel('timestamp in a week')
plt.ylabel('minutes in a hour')
plt.plot(x_data[:, 1], y_data[:, 0], 'r+')
# Plot the hypothesis.
predict_x = np.linspace(0, 168, 1000)
predict_y = w_data[0, 0] + w_data[0, 1] * np.sin(
    w_data[0, 2] + w_data[0, 3] * predict_x)
plt.plot(predict_x, predict_y)
plt.show()
