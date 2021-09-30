import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = nn.BCELoss(size_average=True)
optimize = torch.optim.SGD(model.parameters(), lr=0.01)

l_loss = []
for epoch in range(1000):
    # Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch:', epoch, ' loss:', loss.item())
    l_loss.append(loss.item())
    # Backward
    optimize.zero_grad()
    loss.backward()

    # Update
    optimize.step()

plt.plot(range(1000), l_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

