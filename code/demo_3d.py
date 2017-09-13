import numpy as np
import pylab as plt


import GPy

from util import plot_with_uncertainty
import ep_unimodality as ep
from importlib import reload
reload(ep)

import unimodal 
reload(unimodal)

####################################################################################################################################################3
# Parameters and settings
####################################################################################################################################################3

# set seed
np.random.seed(110)

# dimension
D = 3

####################################################################################################################################################3
# Define test objective function
####################################################################################################################################################3
f = lambda x, y, z: 0.1*((x-2)**2 + (y-2)**2 + (z-2)**2)

####################################################################################################################################################3
# Generate initial observations
####################################################################################################################################################3

# sample points
sigma2 = 3.
N = 20
X = np.random.normal(0, 3, size = (N, 3))
y = f(X[:, 0], X[:, 1], X[:, 2])[:, None] + np.random.normal(0, np.sqrt(sigma2), size=(N, 1))

# Generate test set
Ntest = 1000
Xtest = np.random.normal(0, 3, size = (Ntest, 3))
ytest = f(Xtest[:, 0], Xtest[:, 1], Xtest[:, 2])[:, None] + np.random.normal(0, np.sqrt(sigma2), size=(Ntest, 1))

# For prediction and plotting
xs = np.linspace(-10, 10, 21)
ys = np.linspace(-10, 10, 21)
zs = np.linspace(-10, 10, 21)
Xs, Ys, Zs = np.meshgrid(xs, ys, zs)
Xp = np.column_stack((Xs.ravel(), Ys.ravel(), Zs.ravel()))
fp = f(Xs, Ys, Zs)


##############################################################################################################3
# Fit regular GP for reference
##############################################################################################################3

variance = 10
scale = 4
bias = 1

# Build kernel
kernel = GPy.kern.RBF(input_dim=D, lengthscale=scale, variance=variance) + GPy.kern.Bias(input_dim=D, variance=bias)

gpy_model = GPy.models.GPRegression(X=X, Y=y, kernel=kernel, noise_var=sigma2)
gpy_model.Gaussian_noise.fix()
gpy_model.optimize()
mu_gpy, var_gpy = gpy_model.predict(Xp)

gpy_lppd_train = np.mean(gpy_model.log_predictive_density(X, y))
gpy_lppd_test = np.mean(gpy_model.log_predictive_density(Xtest, ytest))

print(100*'-')
print('\tRegular')
print(100*'-')
print('\tLPPD train: %4.3f' % gpy_lppd_train)
print('\tLPPD test: %4.3f\n\n' % gpy_lppd_test)


##############################################################################################################3
# Fit unimodal GP
##############################################################################################################3
g_variance, g_lengthscale = 1., 4.

# prepare f kernel
f_kernel_base = GPy.kern.RBF(input_dim = D, lengthscale=scale, variance=variance) + GPy.kern.Bias(input_dim=D, variance=bias)

# prepare g kernel
g_kernel_base = GPy.kern.RBF(input_dim = D, lengthscale=g_lengthscale, variance=g_variance) #+ GPy.kern.Bias(input_dim=D, variance=c3)

# add priors
g_kernel_base.variance.unconstrain()
g_kernel_base.variance.set_prior(GPy.priors.StudentT(mu=0, sigma=1, nu=4))
g_kernel_base.variance.constrain_positive()


# Define point grid for pseudoobservations
M = 5
x1 = np.linspace(-10, 10, M)
x2 = np.linspace(-10, 10, M)
x3 = np.linspace(-10, 10, M)
X1, X2, X3 = np.meshgrid(x1, x2, x3)
Xd = np.column_stack((X1.ravel(), X2.ravel(), X3.ravel()))

# fit model
unimodal_model = unimodal.UnimodalGP(X=X, Y=y, Xd=Xd, f_kernel_base=f_kernel_base, g_kernel_base=g_kernel_base, sigma2=sigma2)
unimodal_model.optimize(messages=True)
print(unimodal)

# make predictions
mu_ep, var_ep = unimodal_model.predict(Xp)
mu_ep_g1, var_ep_g1 = unimodal_model.predict_g(Xp, g_index=0)
mu_ep_g2, var_ep_g2 = unimodal_model.predict_g(Xp, g_index=1)


# compute LPPD
unimodal_lppd_train = np.mean(unimodal_model.log_predictive_density(X, y))
unimodal_lppd_test = np.mean(unimodal_model.log_predictive_density(Xtest, ytest))

print(100*'-')
print('\tUnimodal')
print(100*'-')
print('\tLPPD train: %4.3f' % unimodal_lppd_train)
print('\tLPPD test: %4.3f\n\n' % unimodal_lppd_test)