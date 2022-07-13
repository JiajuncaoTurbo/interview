import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def mat_1d(n):
    d1 = sp.diags([-1, 1], [-1, 1], shape=(n, n))
    d1 = sp.lil_matrix(d1)
    d1[0, [0, 1, 2]] = [-3, 4, -1]
    d1[n - 1, [n - 3, n - 2, n - 1]] = [1, -4, 3]

    d2 = sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    d2 = sp.lil_matrix(d2)
    d2[0, [0, 1, 2, 3]] = [2, -5, 4, -1]
    d2[n - 1, [n - 4, n - 3, n - 2, n - 1]] = [-1, 4, -5, 2]

    return d1, d2


def mat_2d(Nx, Ny):
    dx1, d2x1 = mat_1d(Nx)
    dy1, d2y1 = mat_1d(Ny)

    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)

    dx_2d = sp.kron(Iy, dx1)
    dy_2d = sp.kron(dy1, Ix)

    d2x_2d = sp.kron(Iy, d2x1)
    d2y_2d = sp.kron(d2y1, Ix)

    return dx_2d.tolil(), dy_2d.tolil(), d2x_2d.tolil(), d2y_2d.tolil()#derivate_variable_dimension


Nx = 100
Ny = 100
x_length = 1
y_length = 1
x = np.linspace(0, x_length, Nx)
y = np.linspace(0, y_length, Ny)

dx = (x[-1] - x[0]) / Nx
dy = (y[-1] - y[0]) / Ny

X, Y = np.meshgrid(x, y)

X1 = X.reshape(-1)
Y1 = Y.reshape(-1)

g = np.zeros([Ny,Nx])
for i in range(Ny):
    for j in range(Nx):
        g[i, j] = -4 * (-np.power(Y[i, j], 2) + Y[i, j]) * np.sin(np.pi * X[i, j])
g = g.reshape(-1, 1)

index_boundary = np.squeeze(np.where((X1 == x[0]) | (X1 == x[Nx - 1]) | (Y1 == y[0]) | (Y1 == y[Ny - 1])))

dx_2d, dy_2d, d2x_2d, d2y_2d = mat_2d(Nx, Ny)

I = sp.eye(Nx * Ny).tolil()
A = d2x_2d / dx ** 2 + d2y_2d / dy ** 2

A[index_boundary, :] = I[index_boundary, :]

b = g
b[index_boundary] = 0

u = spsolve(A, b).reshape(Ny, Nx)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(x, y, u)
fig.colorbar(cp)
ax.set_title('U Distribution')
plt.show()
