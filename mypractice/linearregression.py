import torch
import torch.nn as nn
import matplotlib.pyplot as plt


x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        pred = self.linear(x)
        return pred


model = LinearModel()

criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

l_loss = []
for e in range(200):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)     # 前馈
    print('epoch:', e, ' loss:', loss.item())
    l_loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()       # 反馈

    optimizer.step()     # 更新权重

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)

print('y_pred:', y_test)

plt.plot(range(200), l_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


