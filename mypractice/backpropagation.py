import torch
import matplotlib.pyplot as plt


x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x*w


def loss(x_, y_):
    y_hat = forward(x_)
    return (y_hat - y_) ** 2


l_loss = []
for e in range(100):
    for x_d, y_d in zip(x_data, y_data):
        l = loss(x_d, y_d)
        l.backward()
        print('x:', x_d, ',y:', y_d, ',w gradient:', w.grad.item())
        w.data = w.data - 0.01*w.grad.data
        w.grad.data.zero_()
    l_loss.append(l.item())
    print('epoch:',e, ' ', l.item())
print('prediction x=4, y=', forward(4).item())
plt.plot(range(100), l_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


