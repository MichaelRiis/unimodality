import autograd.numpy as np
import pylab as plt

from autograd.scipy.stats import norm, t, multivariate_normal as mvn

import GPy

from util import plot_with_uncertainty
import ep_unimodality_2d as ep
from importlib import reload
reload(ep)

# auxilary functions
phi = lambda x: norm.cdf(x)
npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))




####################################################################################################################################################3
# Parameters
####################################################################################################################################################3

# set seed
np.random.seed(110)

# dimension
D = 3



####################################################################################################################################################3
# Define test objective function
####################################################################################################################################################3

rho = .9
K = np.zeros((D, D))
for i in range(D):
	for j in range(D):
		K[i,j] = rho**np.abs(i-j) + 1e-6*(i == j)

L = np.linalg.cholesky(K)
mu = np.array([1, 2, 3])

# def f(x):
	# use negative multivariate normal as test objective
	# b = np.linalg.solve(L, (x - mu).T)
	# return -(-0.5*D*np.log(2*np.pi) - np.sum(np.log(np.diag(L))) - 0.5*np.sum(b**2, axis = 0))

def f(x):
	return np.sum(np.atleast_2d(x - mu)**2, axis = 1)

f0 = f(mu)


####################################################################################################################################################3
# Generate initial observations
####################################################################################################################################################3
N = 20
X = np.random.normal(0, 3, size = (N, D))

sigma2 = 1e-1
y = f(X) + np.random.normal(0, np.sqrt(sigma2), size = (N))


####################################################################################################################################################3
# GPy
####################################################################################################################################################3

# Grid for prediction
Q = 11
xs = np.linspace(-10, 10, Q)
ys = np.linspace(-10, 10, Q)
zs = np.linspace(-10, 10, Q)
Xp, Yp, Zp = np.meshgrid(xs, ys, zs)
XYZ = np.column_stack((Xp.ravel(), Yp.ravel(), Zp.ravel()))
fp = f(XYZ)


# Build kernel
lengthscale = 10.
variance = 100.
rbf = GPy.kern.RBF(input_dim=3, lengthscale=lengthscale, variance=variance)


# fit initial model
gpy_model = GPy.models.GPRegression(X=X, Y=y[:, None], kernel=rbf, noise_var=sigma2)

# make predictions
mu_gpy, var_gpy = gpy_model.predict_noiseless(Xnew=XYZ)



####################################################################################################################################################3
# Unimodal
####################################################################################################################################################3
# Fit Unimodal GP
M = 12
x1 = np.linspace(-10, 10, M)
x2 = np.linspace(-10, 10, M)
x3 = np.linspace(-10, 10, M)
X1, X2, X3 = np.meshgrid(x1, x2, x3)
Xd = np.column_stack((X1.ravel(), X2.ravel(), X3.ravel()))

# fit initial model
mu_f, Sigma_f, Sigma_full_f, g_posterior_list, Kf = ep.ep_unimodality(X, y[:, None], k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, t2=Xd, verbose=10, nu2=1., c1 = 1.)

# make predictions
mu_ep, var_ep = ep.predict(mu_f, Sigma_full_f, X, [Xd, Xd, Xd], XYZ, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2)
# mu_ep_g1, var_ep_g1 = ep.predict(g_posterior_list[0][0], g_posterior_list[0][2], Xd, [Xd, None, None], XYZ, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)
# mu_ep_g2, var_ep_g2 = ep.predict(g_posterior_list[1][0], g_posterior_list[0][2], Xd, [None, Xd, None], XYZ, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)
# mu_ep_g3, var_ep_g3 = ep.predict(g_posterior_list[2][0], g_posterior_list[0][2], Xd, [None, None, Xd], XYZ, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)



####################################################################################################################################################3
# Evaluation
####################################################################################################################################################3
print(2*'\n')

# GPY
err_gpy = np.mean((fp - mu_gpy.ravel())**2)/np.mean(fp**2)
print('GPY: {}'.format(err_gpy))


# unimodal
err_uni = np.mean((fp - mu_ep.ravel())**2)/np.mean(fp**2)
print('Uni: {}'.format(err_uni))


