import matplotlib.pyplot as plt
import numpy


x_trian = [1.0,2.0,3.0]
y_train = [2.0,4.0,6.0]

w = 1.0    # init weight
lr = 0.01   # learning rate
epoch = 100


def forward(x):
    return x*w


def gradient(xs, ys):
    gra = 0
    for xi, yi in zip(xs, ys):
        gra += 2 * xi * (xi * w - yi)
    return gra / len(xs)


def loss(xs, ys):
    cost = 0
    for xi, yi in zip(xs, ys):
        p_pred = forward(xi)
        cost += (p_pred - yi)**2
    return cost / len(xs)


cost_ = []
for e in range(epoch):
    cost_val = loss(x_trian, y_train)
    gradient_val = gradient(x_trian, y_train)
    w -= lr * gradient_val
    cost_.append(cost_val)
    print('epoch:{0}, cost:{1}, w:{2}'.format(e, cost_val, w))

print('predict when x=4, y={}'.format(forward(4)))

plt.plot(range(epoch), cost_)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

