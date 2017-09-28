import numpy as np
import pylab as plt

from importlib import reload

import GPy

from util import plot_with_uncertainty
from ep_unimodality import phi

import unimodal 
reload(unimodal)

####################################################################################################################################################3
# Parameters and settings
####################################################################################################################################################3

# set seed
np.random.seed(110)

# dimension
D = 1

####################################################################################################################################################3
# Define test objective function
####################################################################################################################################################3
mu = 3
def f(x):
    return 0.1*(x-mu)**2

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
# gpy_model.Gaussian_noise.fix()	
gpy_model.optimize()

# make predictions
mu_gpy, var_gpy = gpy_model.predict(Xnew=Xp)

##############################################################################################################3
# Fit unimodal GP
##############################################################################################################3
g_variance, g_lengthscale = 1., 1.

# prepare f kernel
f_kernel_base = GPy.kern.RBF(input_dim = D, lengthscale=scale, variance=variance) + GPy.kern.Bias(input_dim=D, variance=bias)

# prepare g kernel
g_kernel_base = GPy.kern.RBF(input_dim = D, lengthscale=g_lengthscale, variance=g_variance)

# add prior to magnitude of g
g_kernel_base.variance.unconstrain()
g_kernel_base.variance.set_prior(GPy.priors.StudentT(mu=0, sigma=1, nu=4))
g_kernel_base.variance.constrain_positive()

# Define grid points for pseudoobservations
M = 20
Xd = np.linspace(-12, 12, M)[:, None]

# fit model
unimodal_model = unimodal.UnimodalGP(X=X, Y=y, Xd=Xd, f_kernel_base=f_kernel_base, g_kernel_base=g_kernel_base, likelihood=GPy.likelihoods.Gaussian(variance=sigma2))
unimodal_model.optimize(messages=True)
print(unimodal)

# make predictions
mu_ep, var_ep = unimodal_model.predict(Xp)
mu_g_pred, sigma_g_pred = unimodal_model.predict_g(Xp, g_index=0) #(mu_g, Sigma_full_g, Xd, [Xd], Xp, g_kernel_base.variance, g_kernel_base.lengthscale)


##############################################################################################################3
# plot
##############################################################################################################3
fig = plt.figure(figsize = (15, 5))
plt.subplot(1,3, 1)
plot_with_uncertainty(Xp, mu_ep, yvar=var_ep, color='r', label = 'Unimodal')
plot_with_uncertainty(Xp.ravel(), mu_gpy.ravel(), yvar=var_gpy.ravel(), color='b', label = 'Regular')
plt.plot(X, y, 'k.', label = 'Data')
plt.plot(Xp, f(Xp), 'g-')
plt.grid(True)
plt.legend()
plt.ylim((-3, 20))
plt.title('Function f')

plt.subplot(1, 3, 2)
pz = phi(mu_g_pred/np.sqrt(1 + sigma_g_pred))
pz_mean, pz_var = unimodal_model.sample_z_probabilities(Xp, g_index=0)
plot_with_uncertainty(x=Xp, y=pz_mean, yvar=pz_var)
plt.axvline(mu, color = 'k', linestyle='--', alpha=0.5)
plt.ylim((-0.3, 1.3))
plt.grid(True)
plt.title('Post. probabilities of sign of grad')

plt.subplot(1, 3, 3)
plot_with_uncertainty(x=Xp, y=mu_g_pred, yvar=sigma_g_pred)
plt.grid(True)
plt.axvline(mu, color = 'k', linestyle='--', alpha=0.5)
plt.title('Latent function g')
fig.tight_layout()

plt.show()
