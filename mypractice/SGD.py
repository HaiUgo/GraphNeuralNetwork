import matplotlib.pyplot as plt
import numpy


x_trian = [1.0,2.0,3.0]
y_train = [2.0,4.0,6.0]

w = 1.0    # init weight
lr = 0.01   # learning rate
epoch = 100


def forward(x):
    return x*w


def gradient(xi, yi):
    return 2 * xi * (xi * w - yi)


def loss(xi, yi):
    p_pred = forward(xi)
    return (p_pred - yi)**2


cost_ = []
for e in range(epoch):
    for xi, yi in zip(x_trian, y_train):
        cost_val = loss(xi, yi)
        gradient_val = gradient(xi, yi)
        w -= lr * gradient_val
    cost_.append(cost_val)
    print('epoch:{0}, cost:{1}, w:{2}'.format(e, cost_val, w))

print('predict when x=4, y={}'.format(forward(4)))

plt.plot(range(epoch), cost_)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

