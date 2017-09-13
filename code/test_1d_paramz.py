import numpy as np
import pylab as plt

from scipy.stats import norm, t, multivariate_normal as mvn

import GPy

from util import plot_with_uncertainty
import ep_unimodality_2d as ep
from importlib import reload
reload(ep)

import unimodal 
reload(unimodal)


####################################################################################################################################################3
# Auxilary functions
####################################################################################################################################################3
phi = lambda x: norm.cdf(x)


####################################################################################################################################################3
# Parameters and settings
####################################################################################################################################################3

# set seed
np.random.seed(110)

# dimension
D = 1

# optimization tol
tol = 1e-8


####################################################################################################################################################3
# Define test objective function
####################################################################################################################################################3
mu = 3
def f(x):
    return 0.1*(x-mu)**2

f0 = f(mu)


####################################################################################################################################################3
# Generate initial observations
####################################################################################################################################################3
sigma2 = 3

# training set
N = 30
X = np.random.normal(0, 3, size = (N, D))
y = f(X) + np.random.normal(0, np.sqrt(sigma2), size = (N, 1))

# for predctions
Xp = np.linspace(-12, 12, 1001)[:, None]


##############################################################################################################3
# Fit regular GP for reference
##############################################################################################################3

variance = 5
scale = 2
bias = 1.

# Build kernel
kernel = GPy.kern.RBF(input_dim=D, lengthscale=scale, variance=variance) + GPy.kern.Bias(input_dim=D, variance=bias)

# fit initial model
gpy_model = GPy.models.GPRegression(X=X, Y=y, kernel=kernel, noise_var=sigma2)
gpy_model.optimize()

# make predictions
mu_gpy, var_gpy = gpy_model.predict(Xnew=Xp)

##############################################################################################################3
# Fit unimodal GP
##############################################################################################################3
c1, c2 = 1., 1.

# prepare f kernel
f_kernel_base = GPy.kern.RBF(input_dim = D, lengthscale=scale, variance=variance) + GPy.kern.Bias(input_dim=D, variance=bias)

# add priors
# f_kernel_base.parameters[0].variance.unconstrain()
# f_kernel_base.parameters[0].variance.set_prior(GPy.priors.Gamma.from_EV(1, 100))
# f_kernel_base.parameters[0].variance.constrain_positive()

# f_kernel_base.parameters[1].variance.unconstrain()
# f_kernel_base.parameters[1].variance.set_prior(GPy.priors.Gamma.from_EV(1, 100))
# f_kernel_base.parameters[1].variance.constrain_positive()

# prepare g kernel
g_kernel_base = GPy.kern.RBF(input_dim = D, lengthscale=c2, variance=c1) #+ GPy.kern.Bias(input_dim=D, variance=c3)

# add priors
g_kernel_base.variance.unconstrain()
g_kernel_base.variance.set_prior(GPy.priors.StudentT(mu=0, sigma=1, nu=4))
g_kernel_base.variance.constrain_positive()


# Define point grid for pseudoobservations
M = 20
Xd = np.linspace(-12, 12, M)[:, None]

# fit model
unimodal_model = unimodal.UnimodalGP(X=X, Y=y, Xd=Xd, f_kernel_base=f_kernel_base, g_kernel_base=g_kernel_base, sigma2=sigma2)
unimodal_model.optimize(messages=True)
print(unimodal)

# make predictions
mu_ep, var_ep = unimodal_model.predict(Xp)
mu_g_pred, sigma_g_pred = unimodal_model.predict_g(Xp, g_index=0) #(mu_g, Sigma_full_g, Xd, [Xd], Xp, g_kernel_base.variance, g_kernel_base.lengthscale)

# make predictions
mu_g, Sigma_g, Sigma_full_g, Lg = unimodal_model.g_posterior_list[0]


##############################################################################################################3
# plot
##############################################################################################################3

fig = plt.figure(figsize = (15, 5))
plt.subplot(1,3, 1)
plot_with_uncertainty(Xp, mu_ep, ystd=np.sqrt(var_ep), color='r', label = 'Unimodal')
plot_with_uncertainty(Xp.ravel(), mu_gpy.ravel(), ystd=np.sqrt(var_gpy.ravel()), color='b', label = 'Regular')
plt.plot(X, y, 'k.', label = 'Data')
plt.plot(Xp, f(Xp), 'g-')
plt.grid(True)
plt.legend()
plt.ylim((-3, 20))


plt.subplot(1, 3, 2)
pz = phi(mu_g_pred/np.sqrt(1 + sigma_g_pred))
pz_mean, pz_var = ep.sample_z_probabilities(mu_g, Sigma_full_g, Xd, [Xd], Xp, c1, c2)
plot_with_uncertainty(x=Xp, y=pz_mean, ystd=np.sqrt(pz_var))
plt.axvline(mu, color = 'k', linestyle='--', alpha=0.5)
plt.ylim((-0.3, 1.3))
plt.grid(True)

plt.subplot(1, 3, 3)
plot_with_uncertainty(x=Xp, y=mu_g_pred, ystd=np.sqrt(sigma_g_pred))
plt.grid(True)
plt.ylim((-12, 12))
plt.axvline(mu, color = 'k', linestyle='--', alpha=0.5)
fig.tight_layout()

plt.show()
