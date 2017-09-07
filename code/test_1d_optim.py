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
sigma2 = 3

# training set
N = 20
X = np.random.normal(0, 3, size = (N, D))
y = f(X) + np.random.normal(0, np.sqrt(sigma2), size = (N, 1))

Xp = np.linspace(-10, 10, 1001)[:, None]


# test set
Ntest = 1000
Xtest = np.random.normal(0, 3, size = (Ntest, D))
ytest = f(Xtest) + np.random.normal(0, np.sqrt(sigma2), size = (Ntest, 1))

logzs = []
scale_grad = []
scales = np.logspace(-1, 3, 51)


variance = 100.
scale = 1.
##############################################################################################################3

# Build kernel
rbf = GPy.kern.RBF(input_dim=D, lengthscale=scale, variance=variance)


# fit initial model
gpy_model = GPy.models.GPRegression(X=X, Y=y, kernel=rbf, noise_var=sigma2)
gpy_model.optimize()

# make predictions
mu_gpy, var_gpy = gpy_model.predict(Xnew=Xp)

##############3
params = np.array([variance, scale])
log_params = np.log(params)

step_size = 1e-2

# Fit Unimodal GP
M = 20
Xd = np.linspace(-10, 10, M)[:, None]

fig = plt.figure()
for itt in range(20):

	fig.clf()

	variance, scale = np.exp(log_params)

	# fit initial model
	mu_f, Sigma_f, Sigma_full_f, g_posterior_list, Kf, logz_uni, grads = ep.ep_unimodality(X, y, k1=np.sqrt(variance), k2=scale, sigma2=sigma2, t2=Xd, verbose=0, nu2=1., c1 = 1.,  tol=1e-6, max_itt=100)

	# make predictions
	mu_ep, var_ep = ep.predict(mu_f, Sigma_full_f, X, [Xd], Xp, k1=np.sqrt(variance), k2=scale, sigma2=sigma2)

	plot_with_uncertainty(Xp, mu_ep, ystd=np.sqrt(var_ep), color='r', label = 'Unimodal')
	plot_with_uncertainty(Xp.ravel(), mu_gpy.ravel(), ystd=np.sqrt(var_gpy.ravel()), color='b', label = 'Regular')
	plt.plot(X, y, 'k.', label = 'Data')
	plt.plot(Xp, f(Xp), 'g-')
	plt.grid(True)
	plt.legend()
	plt.title('Itt %d' % (itt + 1))
	plt.ylim((-10, 18))

	plt.pause(1e-6)

	if(itt == 0):
		plt.pause(1e-3)
	plt.show(block = False)

	print('Itt %d: Grad norm = %5.4f' % (itt+1, np.linalg.norm(grads)))

	# grad step
	log_params += step_size*grads

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
