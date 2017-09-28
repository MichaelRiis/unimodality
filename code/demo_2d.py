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
D = 2

####################################################################################################################################################3
# Define test objective function
####################################################################################################################################################3
f = lambda x, y: 0.1*((x-2)**2 + (y-2)**2 - 1.2*(x-2)*(y-2))

####################################################################################################################################################3
# Generate initial observations
####################################################################################################################################################3

# sample points
sigma2 = 3.
N = 20
X = np.random.normal(0, 3, size = (N, 2))
y = f(X[:, 0], X[:, 1])[:, None] + np.random.normal(0, np.sqrt(sigma2), size=(N, 1))

# For prediction and plotting
xs = np.linspace(-10, 10, 41)
ys = np.linspace(-10, 10, 41)
Xs, Ys = np.meshgrid(xs, ys)
Xp = np.column_stack((Xs.ravel(), Ys.ravel()))
fp = f(Xs, Ys)


##############################################################################################################3
# Fit regular GP for reference
##############################################################################################################3

variance = 10
scale = 4
bias = 1

# Build kernel
kernel = GPy.kern.RBF(input_dim=D, lengthscale=scale, variance=variance) + GPy.kern.Bias(input_dim=D, variance=bias)

if N > 0:
    gpy_model = GPy.models.GPRegression(X=X, Y=y, kernel=kernel, noise_var=sigma2)
    gpy_model.optimize()
    mu_gpy, var_gpy = gpy_model.predict(Xp)
    mu_gpy = mu_gpy.reshape((len(xs), len(ys)))
    var_gpy = var_gpy.reshape((len(xs), len(ys)))

else:
    mu_gpy = np.zeros((len(xs), len(ys)))
    var_gpy = variance*np.ones((len(xs), len(ys)))


##############################################################################################################3
# Fit unimodal GP
##############################################################################################################3
c1, c2 = 1., 4.

# prepare f kernel
f_kernel_base = GPy.kern.RBF(input_dim = D, lengthscale=scale, variance=variance) + GPy.kern.Bias(input_dim=D, variance=bias)

# add priors
f_kernel_base.parameters[0].variance.unconstrain()
f_kernel_base.parameters[0].variance.set_prior(GPy.priors.StudentT(mu=0, sigma=4, nu=4))
f_kernel_base.parameters[0].variance.constrain_positive()

f_kernel_base.parameters[1].variance.unconstrain()
f_kernel_base.parameters[1].variance.set_prior(GPy.priors.StudentT(mu=0, sigma=4, nu=4))
f_kernel_base.parameters[1].variance.constrain_positive()


# prepare g kernel
g_kernel_base = GPy.kern.RBF(input_dim = D, lengthscale=c2, variance=c1) #+ GPy.kern.Bias(input_dim=D, variance=c3)

# add priors
g_kernel_base.variance.unconstrain()
g_kernel_base.variance.set_prior(GPy.priors.StudentT(mu=0, sigma=1, nu=4))
g_kernel_base.variance.constrain_positive()


# Define point grid for pseudoobservations
M = 10
x1 = np.linspace(-10, 10, M)
x2 = np.linspace(-10, 10, M)
X1, X2 = np.meshgrid(x1, x2)
Xd = np.column_stack((X1.ravel(), X2.ravel()))

# fit model
unimodal_model = unimodal.UnimodalGP(X=X, Y=y, Xd=Xd, f_kernel_base=f_kernel_base, g_kernel_base=g_kernel_base, likelihood=GPy.likelihoods.Gaussian(variance=sigma2))
unimodal_model.optimize(messages=True)
print(unimodal)

# make predictions
mu_ep, var_ep = unimodal_model.predict(Xp)
mu_ep_g1, var_ep_g1 = unimodal_model.predict_g(Xp, g_index=0)
mu_ep_g2, var_ep_g2 = unimodal_model.predict_g(Xp, g_index=1)

# reshape to 2D
mu_ep = mu_ep.reshape((len(xs), len(ys)))
var_ep = var_ep.reshape((len(xs), len(ys)))
mu_ep_g1 = mu_ep_g1.reshape((len(xs), len(ys)))
mu_ep_g2 = mu_ep_g2.reshape((len(xs), len(ys)))

var_ep_g1 = var_ep_g1.reshape((len(xs), len(ys)))
var_ep_g2 = var_ep_g2.reshape((len(xs), len(ys)))


####################################################################################################################################################3
# Plot
####################################################################################################################################################3


# for cross section plot
x0, y0 = 0, 0
idx2 = np.argmin((xs-x0)**2)


# plot
fig = plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.contourf(xs, ys, f(Xs, Ys), 30)
plt.axhline(y0, color='k', linestyle='--')
plt.plot(X[:, 0], X[:, 1], 'r.')
plt.colorbar()
plt.title('True functions and observations')

plt.subplot(2, 3, 2)
plt.contourf(xs, ys, mu_gpy, 30)
plt.plot(X[:, 0], X[:, 1], 'r.')
plt.colorbar()
plt.title('Regular GP')

plt.subplot(2, 3, 3)
plt.contourf(xs, ys, mu_ep, 30)
plt.plot(X[:, 0], X[:, 1], 'r.')
plt.colorbar()
plt.title('Unimodal GP')

# cross section
plt.subplot(2, 3, 4)
plt.plot(xs, f(xs, y0), label = 'True function', color='g')
plot_with_uncertainty(xs,  mu_ep[idx2, :], yvar=var_ep[idx2, :], color='r', label='Unimodal', linestyle='--')
plot_with_uncertainty(xs,  mu_gpy[idx2, :], yvar=var_gpy[idx2, :], color='b', label='Regular')
plt.legend()
plt.grid(True)
plt.title('Cross section plot')

if M > 1:

    plt.subplot(2, 3, 5)
    mu_g = unimodal_model.g_posterior_list[0].mean[:(len(x1)*len(x2))].reshape((len(x1), len(x2)))
    plt.contourf(xs, xs, ep.phi(mu_ep_g1/np.sqrt(1+var_ep_g1)), 30)
    plt.plot(Xd[:, 0], Xd[:, 1], 'k.', markersize = 2, alpha=0.75)
    plt.colorbar()
    plt.title('Posterior mean deriv1')

    plt.subplot(2, 3, 6)
    plt.contourf(xs, ys, ep.phi(mu_ep_g2/np.sqrt(1+var_ep_g2)), 30)
    plt.plot(Xd[:, 0], Xd[:, 1], 'k.', markersize = 2, alpha=0.75)
    plt.colorbar()
    plt.title('Posterior mean deriv2')


fig.tight_layout()
plt.show()





