import autograd.numpy as np
import pylab as plt

from autograd.scipy.stats import norm, t, multivariate_normal as mvn

import GPy

from util import plot_with_uncertainty
import ep_unimodality_2d as ep
from importlib import reload
reload(ep)


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

f0 = f(mu)


####################################################################################################################################################3
# Generate initial observations
####################################################################################################################################################3
sigma2 = 1.

# training set
N = 10
X = np.random.normal(0, 3, size = (N, D))
y = f(X) + np.random.normal(0, np.sqrt(sigma2), size = (N, 1))


# test set
Ntest = 1000
Xtest = np.random.normal(0, 3, size = (Ntest, D))
ytest = f(Xtest) + np.random.normal(0, np.sqrt(sigma2), size = (Ntest, 1))

####################################################################################################################################################3
# GPy
####################################################################################################################################################3

# Grid for prediction
Q = 101
Xp = np.linspace(-10, 10, Q)[:, None]
fp = f(Xp)


# Build kernel
lengthscale = 10.
variance = 100.
rbf = GPy.kern.RBF(input_dim=D, lengthscale=lengthscale, variance=variance)


# fit initial model
gpy_model = GPy.models.GPRegression(X=X, Y=y, kernel=rbf, noise_var=sigma2)

# make predictions
mu_gpy, var_gpy = gpy_model.predict(Xnew=Xp)

# evaluate
lppd_gpy = np.mean(gpy_model.log_predictive_density(Xtest, ytest))
err_gpy = np.mean((fp.ravel() - mu_gpy.ravel())**2)/np.mean(fp.ravel()**2)
logz_gpy = -gpy_model.objective_function()


####################################################################################################################################################3
# Unimodal
####################################################################################################################################################3
# Fit Unimodal GP
M = 20
Xd = np.linspace(-10, 10, M)[:, None]

# fit initial model
mu_f, Sigma_f, Sigma_full_f, g_posterior_list, Kf, logz_uni = ep.ep_unimodality(X, y, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, t2=Xd, verbose=10, nu2=1., c1 = 1.)

# make predictions
mu_ep, var_ep = ep.predict(mu_f, Sigma_full_f, X, [Xd], Xp, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2)
mu_ep_g1, var_ep_g1 = ep.predict(g_posterior_list[0][0], g_posterior_list[0][2], Xd, [Xd], Xp, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)

# evaluate
lppd_uni = ep.lppd(ytest, mu_f, Sigma_full_f, X, [Xd], Xtest, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2)
err_uni = np.mean((fp.ravel() - mu_ep.ravel())**2)/np.mean(fp.ravel()**2)

####################################################################################################################################################3
# Evaluation
####################################################################################################################################################3
print(2*'\n')

names = ['GPy', 'Uni']
lppds = [lppd_gpy, lppd_uni]
nmses = [err_gpy, err_uni]
logzs = [logz_gpy, logz_uni]

print(60*'-')
print('%10s\t%s\t\t%s\t\t%s' % ('Name', 'LPPD', 'NMSE', 'log Z'))
print(60*'-')

for name, lppd, nmse, logz in zip(names, lppds, nmses, logzs):
	print('%10s\t%4.3f\t\t%4.3f\t\t%4.3f' % (name, lppd, nmse, logz))

####################################################################################################################################################3
# Plot
####################################################################################################################################################3

plt.plot(Xp, fp, 'g-', label='True function')
plt.plot(Xtest, ytest, 'k.', label='Test observations', alpha = 0.15, markersize=2)
plt.plot(X, y, 'k.', label='Training Observations')
plot_with_uncertainty(Xp, mu_ep, np.sqrt(var_ep), color='r', label='Unimodality')
plot_with_uncertainty(Xp.ravel(), mu_gpy.ravel(), ystd=np.sqrt(var_gpy.ravel()), color='b', label='Regular')
plt.xlabel('Input space')
plt.grid(True)
plt.legend()
plt.show()
