import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        pred = F.sigmoid(self.linear(x))
        return pred


model = LogisticRegressionModel()
criterion = nn.BCELoss(size_average=False)
optimize = torch.optim.SGD(model.parameters(), lr=0.01)

l_loss = []
for e in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    l_loss.append(loss.item())
    print('epoch:', e, ' loss:', loss.item())
    optimize.zero_grad()
    loss.backward()
    optimize.step()

plt.plot(range(1000), l_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

print('prediction x=4', ' y=', model(torch.Tensor([[4.0]])))


x = np.linspace(0, 10, 200)
print(x)
x_t = torch.Tensor(x).view((200, 1))
print(x_t)
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()

