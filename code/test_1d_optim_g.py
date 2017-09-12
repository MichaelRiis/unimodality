import numpy as np
import pylab as plt

from scipy.stats import norm, t, multivariate_normal as mvn

import GPy

from util import plot_with_uncertainty
import ep_unimodality_2d as ep
from importlib import reload
reload(ep)



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


# prepare f kernel
f_kernel = GPy.kern.RBF(input_dim = D, lengthscale=scale, variance=variance) + GPy.kern.Bias(input_dim=D, variance=bias)

# add priors
f_kernel.parameters[0].variance.unconstrain()
f_kernel.parameters[0].variance.set_prior(GPy.priors.Gamma.from_EV(100, 100))
f_kernel.parameters[0].variance.constrain_positive()

f_kernel.parameters[1].variance.unconstrain()
f_kernel.parameters[1].variance.set_prior(GPy.priors.Gamma.from_EV(100, 100))
f_kernel.parameters[1].variance.constrain_positive()

c1, c2 = 1., 1.

params = np.array([variance, scale, bias, c1, c2])
log_params = np.log(params)

step_size = 1e-2

# Fit Unimodal GP
M = 20
Xd = np.linspace(-10, 10, M)[:, None]

Ls = []
gnorm = []
logparams = []

names = ['K1', 'K2', 'K3', 'C1', 'C2']

fig = plt.figure()
for itt in range(500):

	# store old parameters
	old_params = log_params.copy()

	# map current parameter to parameter space
	k1, k2, bias, c1, c2 = np.exp(log_params)

	# update kernel
	f_kernel[:] = [k1, k2, bias]

	# fit model with n ew parameters
	mu_f, Sigma_f, Sigma_full_f, g_posterior_list, Kf, logz_uni, grads = ep.ep_unimodality(X, y, f_kernel=f_kernel, sigma2=sigma2, t2=Xd, verbose=0, nu2=1., c1=c1,  c2=c2, tol=1e-6, max_itt=100)

	# map gradients to log space
	grads *= np.exp(log_params)

	# store current values
	Ls.append(logz_uni)
	gnorm.append(np.linalg.norm(grads))
	logparams.append(log_params.copy())

	# take grad step
	log_params += step_size*grads

	# convergence check
	diff = np.mean((log_params - old_params)**2)/np.mean(old_params**2)
	if diff < tol:
		print('Converged in %d iterations with diff = %4.3e' % (itt, diff))
		break
	else:
		print('Itt %d: Grad norm = %5.4f, diff = %5.4e' % (itt+1, np.linalg.norm(grads), diff))


# make predictions
mu_g, Sigma_g, Sigma_full_g, Lg = g_posterior_list[0]
mu_ep, var_ep = ep.predict(mu_f, Sigma_full_f, X, [Xd], Xp, k1=variance, k2=scale, k3=bias, sigma2=sigma2)
mu_g_pred, sigma_g_pred = ep.predict(mu_g, Sigma_full_g, Xd, [Xd], Xp, c1, c2)



# plot
plt.subplot(2,3, 1)
plot_with_uncertainty(Xp, mu_ep, ystd=np.sqrt(var_ep), color='r', label = 'Unimodal')
plot_with_uncertainty(Xp.ravel(), mu_gpy.ravel(), ystd=np.sqrt(var_gpy.ravel()), color='b', label = 'Regular')
plt.plot(X, y, 'k.', label = 'Data')
plt.plot(Xp, f(Xp), 'g-')
plt.grid(True)
plt.legend()
plt.title('Itt %d' % (itt + 1))
plt.ylim((-10, 20))

plt.subplot(2, 3, 2)
plt.plot(Ls)
plt.xlabel('Iteration')
plt.ylabel('log Z')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(gnorm)
plt.xlabel('Iteration')
plt.ylabel('Norm of gradient')
plt.grid(True)

plt.subplot(2, 3, 6)
for idx, values in enumerate(np.array(logparams).T):
	plt.plot(values, label = names[idx])
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Log params')
plt.grid(True)


plt.subplot(2, 3, 4)
pz = phi(mu_g_pred/np.sqrt(1 + sigma_g_pred))
pz_mean, pz_var = ep.sample_z_probabilities(mu_g, Sigma_full_g, Xd, [Xd], Xp, c1, c2)
plot_with_uncertainty(x=Xp, y=pz_mean, ystd=np.sqrt(pz_var))
plt.axvline(mu, color = 'k', linestyle='--', alpha=0.5)
plt.ylim((-0.3, 1.3))
plt.grid(True)

plt.subplot(2, 3, 5)
plot_with_uncertainty(x=Xp, y=mu_g_pred, ystd=np.sqrt(sigma_g_pred))
plt.grid(True)
plt.ylim((-12, 12))
plt.axvline(mu, color = 'k', linestyle='--', alpha=0.5)
fig.tight_layout()



plt.show()

# mu_ep_g1, var_ep_g1 = ep.predict(g_posterior_list[0][0], g_posterior_list[0][2], Xd, [Xd], Xp, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)

# # evaluate
# lppd_uni = ep.lppd(ytest, mu_f, Sigma_full_f, X, [Xd], Xtest, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2)
# err_uni = np.mean((fp.ravel() - mu_ep.ravel())**2)/np.mean(fp.ravel()**2)

# ####################################################################################################################################################3
# # Evaluation
# ####################################################################################################################################################3
# print(2*'\n')

# names = ['GPy', 'Uni']
# lppds = [lppd_gpy, lppd_uni]
# nmses = [err_gpy, err_uni]
# logzs = [logz_gpy, logz_uni]

# print(60*'-')
# print('%10s\t%s\t\t%s\t\t%s' % ('Name', 'LPPD', 'NMSE', 'log Z'))
# print(60*'-')

# for name, lppd, nmse, logz in zip(names, lppds, nmses, logzs):
# 	print('%10s\t%4.3f\t\t%4.3f\t\t%4.3f' % (name, lppd, nmse, logz))

# ####################################################################################################################################################3
# # Plot
# ####################################################################################################################################################3

# plt.plot(Xp, fp, 'g-', label='True function')
# plt.plot(Xtest, ytest, 'k.', label='Test observations', alpha = 0.15, markersize=2)
# plt.plot(X, y, 'k.', label='Training Observations')
# plot_with_uncertainty(Xp, mu_ep, np.sqrt(var_ep), color='r', label='Unimodality')
# plot_with_uncertainty(Xp.ravel(), mu_gpy.ravel(), ystd=np.sqrt(var_gpy.ravel()), color='b', label='Regular')
# plt.xlabel('Input space')
# plt.grid(True)
# plt.legend()
# plt.show()
