import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1234)
np.random.seed(1234)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = torch.nn.ModuleList([nn.Linear(50, 50) for i in range(20)])
        self.act = torch.nn.ModuleList([nn.Tanh() for i in range(20)])
        self.in_layer = nn.Linear(2, 50)
        self.out_layer = nn.Linear(50, 1)

    def forward(self, input):
        input = self.in_layer(input)
        for i in range(20):
            x = self.linear[i](input)
            input = self.act[i](x + input)
        output = self.out_layer(input)

        return output


def f(net, x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    input = torch.cat([x, y], axis=1)
    u = net(input)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

    con = u_xx + u_yy + 4 * (-torch.pow(y, 2) + y) * torch.sin(np.pi * x)

    loss_pde = (con ** 2).mean()

    return loss_pde, u


def f_bc(net, bc):
    out = net(bc)
    loss_bc = (out ** 2).mean()

    return loss_bc


net = Net()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.01)
# scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)

nx = 100
ny = 100
meshx = np.zeros([nx, ny])
meshy = np.zeros([nx, ny])
for i in range(nx):
    for j in range(ny):
        meshx[i, j] = i / nx
        meshy[i, j] = j / ny

bc_left = np.concatenate([meshx[0, :].reshape(-1, 1), meshy[0, :].reshape(-1, 1)], axis=1)
bc_right = np.concatenate([meshx[nx - 1, :].reshape(-1, 1), meshy[nx - 1, :].reshape(-1, 1)], axis=1)
bc_bottom = np.concatenate([meshx[1:nx, 0].reshape(-1, 1), meshy[1:nx, 0].reshape(-1, 1)], axis=1)
bc_up = np.concatenate([meshx[1:nx, ny - 1].reshape(-1, 1), meshy[1:nx, ny - 1].reshape(-1, 1)], axis=1)

bc = np.concatenate([bc_left, bc_right, bc_bottom, bc_up], axis=0)
bc = Variable(torch.from_numpy(bc).float(), requires_grad=True).to(device)

NoT = int(0.95 * nx * ny)
iterations = 20000
previous_validation_loss = 99999999.0

for epoch in range(iterations):
    optimizer.zero_grad()

    idx = np.random.choice(nx, NoT)
    jdx = np.random.choice(ny, NoT)

    x = Variable(torch.from_numpy(meshx[idx, jdx]).float(), requires_grad=True).to(device)
    y = Variable(torch.from_numpy(meshy[idx, jdx]).float(), requires_grad=True).to(device)

    loss_pde, u, = f(net, x, y)
    loss_bc = f_bc(net, bc)
    loss = loss_pde + loss_bc

    loss.backward()
    optimizer.step()

    print(f'epoch {epoch} loss {loss.item():.8f},loss_pde:{loss_pde.item():.8f}, loss_bc:{loss_bc.item():.8f}')
    # scheduler.step()
    # if loss_pde.item()<=0.01 and loss_bc.item()<=0.01:
    #     break

x = Variable(torch.from_numpy(meshx).float(), requires_grad=False).to(device)
y = Variable(torch.from_numpy(meshy).float(), requires_grad=False).to(device)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
input = torch.cat([x, y], axis=1)

u = net(input)
u = u.detach().cpu().numpy()
u = u.reshape(nx, ny)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(meshx, meshy, u)
fig.colorbar(cp)
ax.set_title('U Distribution')
plt.show()
