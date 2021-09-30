import numpy as np
import matplotlib.pyplot as plt

x_train = [1.0,2.0,3.0]
y_train = [2.0,4.0,6.0]


def forward(x):
    return x*w


def loss(y_pre, y_train):
    return (y_pre - y_train)*(y_pre - y_train)


l_w_list = []
l_loss_list = []

for w in np.arange(0.0, 4.1, 0.1):
    l_sum = 0
    for x_val, y_vla in zip(x_train,y_train):
        y_pre = forward(x_val)
        loss_ = loss(y_pre, y_vla)
        l_sum += loss_
    l_w_list.append(w)
    l_loss_list.append(l_sum/3)

plt.title('Demo')
plt.plot(l_w_list, l_loss_list)
plt.xlabel('w')
plt.ylabel('loss')
plt.show()

