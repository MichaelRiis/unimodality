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
np.random.seed(1000)

# dimension
D = 4



####################################################################################################################################################3
# Define test objective function
####################################################################################################################################################3

mu = np.random.normal(0, 1, size =D)

# def f(x):
	# use negative multivariate normal as test objective
	# b = np.linalg.solve(L, (x - mu).T)
	# return -(-0.5*D*np.log(2*np.pi) - np.sum(np.log(np.diag(L))) - 0.5*np.sum(b**2, axis = 0))

def f(x):
	return 1e-1*np.sum(np.atleast_2d(x - mu)**2, axis = 1)

f0 = f(mu)


####################################################################################################################################################3
# Generate initial observations
####################################################################################################################################################3
N = 100
X = np.random.normal(0, 3, size = (N, D))

sigma2 = 1e-1
y = f(X) + np.random.normal(0, np.sqrt(sigma2), size = (N))


# test set
Ntest = 1000
Xtest = np.random.normal(0, 3, size = (Ntest, D))
ytest = f(Xtest)[:, None] + np.random.normal(0, np.sqrt(sigma2), size = (Ntest, 1))



####################################################################################################################################################3
# GPy
####################################################################################################################################################3

# Grid for prediction
Q = 12
xs = np.linspace(-10, 10, Q)
ys = np.linspace(-10, 10, Q)
zs = np.linspace(-10, 10, Q)
ws = np.linspace(-10, 10, Q)
Xp, Yp, Zp, Wp = np.meshgrid(xs, ys, zs, ws)
XYZ = np.column_stack((Xp.ravel(), Yp.ravel(), Zp.ravel(), Wp.ravel()))
fp = f(XYZ)


# Build kernel
lengthscale = 10.
variance = 10.
rbf = GPy.kern.RBF(input_dim=D, lengthscale=lengthscale, variance=variance)


# fit initial model
gpy_model = GPy.models.GPRegression(X=X, Y=y[:, None], kernel=rbf, noise_var=sigma2)

# make predictions
mu_gpy, var_gpy = gpy_model.predict_noiseless(Xnew=XYZ)

# evaluate
lppd_gpy = np.mean(gpy_model.log_predictive_density(Xtest, ytest))
err_gpy = np.mean((fp.ravel() - mu_gpy.ravel())**2)/np.mean(fp.ravel()**2)


####################################################################################################################################################3
# Unimodal
####################################################################################################################################################3

# Fit Unimodal GP
M = 5
x1 = np.linspace(-10, 10, M)
x2 = np.linspace(-10, 10, M)
x3 = np.linspace(-10, 10, M)
x4 = np.linspace(-10, 10, M)
X1, X2, X3, X4 = np.meshgrid(x1, x2, x3, x4)
Xd = np.column_stack((X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel()))

# fit initial model
mu_f, Sigma_f, Sigma_full_f, g_posterior_list, Kf = ep.ep_unimodality(X, y[:, None], k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, t2=Xd, verbose=10, nu2=1.)

# make predictions
mu_ep, var_ep = ep.predict(mu_f, Sigma_full_f, X, [Xd, Xd, Xd, Xd], XYZ, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2)
# mu_ep_g1, var_ep_g1 = ep.predict(g_posterior_list[0][0], g_posterior_list[0][2], Xd, [Xd, None, None], XYZ, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)
# mu_ep_g2, var_ep_g2 = ep.predict(g_posterior_list[1][0], g_posterior_list[0][2], Xd, [None, Xd, None], XYZ, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)
# mu_ep_g3, var_ep_g3 = ep.predict(g_posterior_list[2][0], g_posterior_list[0][2], Xd, [None, None, Xd], XYZ, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)

# evaluate
lppd_uni = ep.lppd(ytest, mu_f, Sigma_full_f, X, [Xd, Xd, Xd, Xd], Xtest, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2)
err_uni = np.mean((fp.ravel() - mu_ep.ravel())**2)/np.mean(fp.ravel()**2)



####################################################################################################################################################3
# Evaluation
####################################################################################################################################################3
print(2*'\n')

names = ['GPy', 'Uni']
lppds = [lppd_gpy, lppd_uni]
nmses = [err_gpy, err_uni]

print(60*'-')
print('%10s\t%s\t\t%s' % ('Name', 'LPPD', 'NMSE'))
print(60*'-')

for name, lppd, nmse in zip(names, lppds, nmses):
	print('%10s\t%4.3f\t\t%4.3f' % (name, lppd, nmse))
