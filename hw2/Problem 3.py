import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# (a)
mean = np.array([1, 1])
cov = np.array([[1, 0],[0, 2]])
var = np.array([])
for i in range(0, cov.shape[0]):
	var = np.append(var, cov[i][i])
maxVar = max(var)

delta = 0.25
x = np.arange(mean[0]-(maxVar+3), mean[0]+(maxVar+3), delta)
y = np.arange(mean[1]-(maxVar+3), mean[1]+(maxVar+3), delta)
X, Y = np.meshgrid(x, y)
Z = mlab.bivariate_normal(X, Y, cov[0][0], cov[1][1], mean[0], mean[1], cov[0][1])

fig = plt.figure()
ax = fig.gca()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
ax.set_aspect('auto')
fig.savefig('a.png')


# (b)
mean = np.array([-1, 2])
cov = np.array([[2, 1],[1, 3]])
var = np.array([])
for i in range(0, cov.shape[0]):
	var = np.append(var, cov[i][i])
maxVar = max(var)

delta = 0.25
x = np.arange(mean[0]-(maxVar+4), mean[0]+(maxVar+4), delta)
y = np.arange(mean[1]-(maxVar+4), mean[1]+(maxVar+4), delta)
X, Y = np.meshgrid(x, y)
Z = mlab.bivariate_normal(X, Y, cov[0][0], cov[1][1], mean[0], mean[1], cov[0][1])

fig = plt.figure()
ax = fig.gca()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
ax.set_aspect('auto')
fig.savefig('b.png')

# (c)
mean1 = np.array([0, 2])
mean2 = np.array([2, 0])
cov1 = np.array([[2, 1],[1, 1]])
cov2 = cov1
for i in range(0, cov.shape[0]):
	var = np.append(var, cov1[i][i])
maxVar = max(var)

delta = 0.25
x = np.arange(1-(maxVar+3), 1+(maxVar+3), delta)
y = np.arange(1-(maxVar+3), 1+(maxVar+3), delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, cov1[0][0], cov1[1][1], mean1[0], mean1[1], cov1[0][1])
Z2 = mlab.bivariate_normal(X, Y, cov2[0][0], cov2[1][1], mean2[0], mean2[1], cov2[0][1])
Z = Z1 - Z2

fig = plt.figure()
ax = fig.gca()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
ax.set_aspect('auto')
fig.savefig('c.png')

# (d)
mean1 = np.array([0, 2])
mean2 = np.array([2, 0])
cov1 = np.array([[2, 1],[1, 1]])
cov2 = np.array([[2, 1],[1, 3]])

delta = 0.25
x = np.arange(1-(maxVar+3), 1+(maxVar+3), delta)
y = np.arange(1-(maxVar+3), 1+(maxVar+3), delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, cov1[0][0], cov1[1][1], mean1[0], mean1[1], cov1[0][1])
Z2 = mlab.bivariate_normal(X, Y, cov2[0][0], cov2[1][1], mean2[0], mean2[1], cov2[0][1])
Z = Z1 - Z2

fig = plt.figure()
ax = fig.gca()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
ax.set_aspect('auto')
fig.savefig('d.png')

# (e)
mean1 = np.array([1, 1])
mean2 = np.array([-1, -1])
cov1 = np.array([[2, 0],[0, 1]])
cov2 = np.array([[2, 1],[1, 2]])

delta = 0.25
x = np.arange(0-(maxVar+3), 0+(maxVar+3), delta)
y = np.arange(0-(maxVar+3), 0+(maxVar+3), delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, cov1[0][0], cov1[1][1], mean1[0], mean1[1], cov1[0][1])
Z2 = mlab.bivariate_normal(X, Y, cov2[0][0], cov2[1][1], mean2[0], mean2[1], cov2[0][1])
Z = Z1 - Z2

fig = plt.figure()
ax = fig.gca()
CS = plt.contour(X, Y, Z)
ax.set_aspect('auto')
fig.savefig('e.png')