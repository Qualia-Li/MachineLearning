import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import math

# Part d)
# Define X and Y data
X = np.array([4, 5, 5.6, 6.8, 7, 7.2, 8, 0.8, 1, 1.2, 2.5, 2.6, 3, 4.3])
Y = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
X.shape = (X.shape[0],1)
Y.shape = (Y.shape[0],1)
# print(X.shape)
# print(Y.shape)

# Standardize X to have mean 0 and variance 1
mean = np.mean(X)
std = np.std(X)
print("The mean of X is:", mean)
print("The standard deviation of X is:", std)

Xstd = (X-mean)/std
print("The mean of Xstd is:", np.mean(Xstd))
print("The variance of Xstd is:", np.var(Xstd))

# Append a column of ones to X vector
ones = np.ones((Xstd.shape[0],1))
Xstd = np.hstack((Xstd, ones))
# print(Xstd.shape)

# Initialize regularization parameter and beta
reg = 0.07
beta = np.array([1, 0])
beta.shape = (beta.shape[0],1)
mu = np.empty([Xstd.shape[0],1])
# print(mu.shape)

# Run Newton's method for logistic regression
for i in range(3):
	for j in range(mu.shape[0]):
		mu[j] = 1/(1+np.exp(-beta.T.dot(Xstd[j,:])))
	# print(mu)
	W = np.diag(mu.reshape(mu.shape[0]))
	# print(W)
	grad = 2*reg*beta-Xstd.T.dot(Y-mu)
	hess = 2*reg*np.identity(beta.shape[0])+Xstd.T.dot(W).dot(Xstd)
	step = la.solve(hess, -grad)
	beta = beta + step
print("Beta for logistic regression is")
print(beta)

x = np.arange(-2,2.1,0.1)
y = 1/(1+np.exp(-beta[0]*x+beta[1]))

# Plot
plt.figure()
plt.plot(Xstd[0:7,0],Xstd[0:7,1],'+',label='y=1')
plt.plot(Xstd[7:14,0],Xstd[7:14,1],'o',label='y=0')
plt.plot(x,y,label='Logistic Regression')

# Run linear regression
beta = np.array([1, 0])
beta.shape = (beta.shape[0],1)

grad = 2*reg*beta-Xstd.T.dot(Y-Xstd.dot(beta))
hess = 2*reg*np.identity(beta.shape[0])+Xstd.T.dot(Xstd)
step = la.solve(hess, -grad)
beta = beta + step
print("Beta for linear regression is")
print(beta)

y = beta[0]*x+beta[1]
plt.plot(x,y,label='Linear Regression')

plt.axis([-2, 2, 0, 1.5])
plt.grid()
plt.legend()

# Part e)
Xstd = np.vstack(([3,1],Xstd))
Y = np.vstack(([1],Y))
# print(Xstd)

# Initialize regularization parameter and beta
reg = 0.07
beta = np.array([1, 0])
beta.shape = (beta.shape[0],1)
mu = np.empty([Xstd.shape[0],1])
# print(mu.shape)

# Run Newton's method for logistic regression
for i in range(3):
	for j in range(mu.shape[0]):
		mu[j] = 1/(1+np.exp(-beta.T.dot(Xstd[j,:])))
	# print(mu)
	W = np.diag(mu.reshape(mu.shape[0]))
	# print(W)
	grad = 2*reg*beta-Xstd.T.dot(Y-mu)
	hess = 2*reg*np.identity(beta.shape[0])+Xstd.T.dot(W).dot(Xstd)
	step = la.solve(hess, -grad)
	beta = beta + step
print("Beta for logistic regression is")
print(beta)

x = np.arange(-2,3.6,0.1)
y = 1/(1+np.exp(-beta[0]*x+beta[1]))

# Plot
plt.figure()
plt.plot(Xstd[0:8,0],Xstd[0:8,1],'+',label='y=1')
plt.plot(Xstd[8:15,0],Xstd[8:15,1],'o',label='y=0')
plt.plot(x,y,label='Logistic Regression')

# Run linear regression
beta = np.array([1, 0])
beta.shape = (beta.shape[0],1)

grad = 2*reg*beta-Xstd.T.dot(Y-Xstd.dot(beta))
hess = 2*reg*np.identity(beta.shape[0])+Xstd.T.dot(Xstd)
step = la.solve(hess, -grad)
beta = beta + step
print("Beta for linear regression is")
print(beta)

y = beta[0]*x+beta[1]
plt.plot(x,y,label='Linear Regression')

plt.title("With Outlier")
plt.axis([-2, 3.5, 0, 1.5])
plt.grid()
plt.legend()

plt.draw()
plt.show()