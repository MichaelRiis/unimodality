import numpy as np
import pylab as plt

from scipy.stats import norm

import GPy

from util import plot_with_uncertainty
import ep_unimodality_2d as ep
from importlib import reload
reload(ep)

phi = lambda x: norm.cdf(x)

# set seed
np.random.seed(0)

# Number of observations
Ns = 2*np.array([0, 10, 20, 30, 40, 50])

# Define test function
f = lambda x, y: 0.1*(x**2 + y**2 + -1.2*x*y)


# For prediction and plotting
xs = np.linspace(-12, 12, 41)
ys = np.linspace(-12, 12, 41)
Xp, Yp = np.meshgrid(xs, ys)
XY = np.column_stack((Xp.ravel(), Yp.ravel()))

# sample points
sigma2 = 1e-1
Nfull = np.max(Ns)
Xfull = np.random.normal(0, 3.5, size = (Nfull, 2))
yfull = f(Xfull[:, 0], Xfull[:, 1])[:, None] + np.random.normal(0, np.sqrt(sigma2), size=(Nfull, 1))

# generate test set
Ntest = 1000
Xtest = np.random.normal(0, 3.5, size = (Ntest, 2))
ytest = f(Xtest[:, 0], Xtest[:, 1])[:, None] + np.random.normal(0, np.sqrt(sigma2), size=(Ntest, 1))

# GP hyperparameters
variance = 10
lengthscale = 3
k3 = 10

perm = np.random.permutation(Nfull)


plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.contourf(xs, ys, f(Xp, Yp), 30)
plt.plot(Xfull[:, 0], Xfull[:, 1], 'r.', alpha=0.5)
plt.colorbar()
plt.title('True functions and training data')

plt.subplot(1, 2, 2)
plt.contourf(xs, ys, f(Xp, Yp), 30)
plt.plot(Xtest[:, 0], Xtest[:, 1], 'g.', alpha=0.5)
plt.colorbar()
plt.title('True functions and test data')

plt.pause(1e-2)
plt.show(block=False)

fig = plt.figure(figsize = (20, 8))
for idx_N, N in enumerate(Ns):


	X = Xfull[perm[:N], :]
	y = yfull[perm[:N], :]


	# Fit regular GP
	kernel = GPy.kern.RBF(input_dim=2, lengthscale=lengthscale, variance=variance) + GPy.kern.Bias(input_dim=2, variance=k3)
	if N > 0:
		model = GPy.models.GPRegression(X=X, Y=y, kernel=kernel, noise_var=sigma2)
		# model.optimize()
		mu, var = model.predict_noiseless(XY)
		mu = mu.reshape((len(xs), len(ys)))
		var = var.reshape((len(xs), len(ys)))

		gpy_lppd = np.mean(model.log_predictive_density(Xtest, ytest))
	else:
		mu = np.zeros((len(xs), len(ys)))
		var = variance*np.ones((len(xs), len(ys)))

		log_npdf = lambda x, m, v: -0.5*np.log(2*np.pi*v) -(x-m)**2/(2*v)
		gpy_lppd = np.mean(log_npdf(ytest, mu.ravel(), var.ravel() + sigma2))



	# Fit Unimodal GP
	M = 10
	x1 = np.linspace(-10, 10, M)
	x2 = np.linspace(-10, 10, M)
	X1, X2 = np.meshgrid(x1, x2)
	Xd = np.column_stack((X1.ravel(), X2.ravel()))


	mu_f, Sigma_f, Sigma_full_f, g_posterior_list, Kf = ep.ep_unimodality(X, y, k1=np.sqrt(variance), k2=lengthscale, k3=k3, sigma2=sigma2, t2=Xd, verbose=10, nu2=1.)
	mu_ep, var_ep = ep.predict(mu_f, Sigma_full_f, X, [Xd, Xd], XY, k1=np.sqrt(variance), k2=lengthscale, k3=k3, sigma2=sigma2)
	mu_ep_g1, var_ep_g1 = ep.predict(g_posterior_list[0][0], g_posterior_list[0][2], Xd, [Xd, None], XY, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)
	mu_ep_g2, var_ep_g2 = ep.predict(g_posterior_list[1][0], g_posterior_list[0][2], Xd, [None, Xd], XY, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)

	mu_ep = mu_ep.reshape((len(xs), len(ys)))
	var_ep = var_ep.reshape((len(xs), len(ys)))
	mu_ep_g1 = mu_ep_g1.reshape((len(xs), len(ys)))
	mu_ep_g2 = mu_ep_g2.reshape((len(xs), len(ys)))

	var_ep_g1 = var_ep_g1.reshape((len(xs), len(ys)))
	var_ep_g2 = var_ep_g2.reshape((len(xs), len(ys)))

	unimodal_lppd = ep.lppd(ytest, mu_f, Sigma_full_f, X, [Xd, Xd], Xtest, k1=np.sqrt(variance), k2=lengthscale, k3=k3, sigma2=sigma2)





	# plot
	# plt.subplot(2, len(Ns), 1 + idx)
	# plt.contourf(xs, ys, f(Xp, Yp), 30)
	# plt.axhline(y0, color='k', linestyle='--')
	# plt.plot(X[:, 0], X[:, 1], 'r.')
	# plt.colorbar()
	# plt.title('True functions and observations')

	plt.subplot(2, len(Ns), 1 + idx_N)
	plt.contourf(xs, ys, mu, 30)
	plt.plot(X[:, 0], X[:, 1], 'r.', alpha=0.5)
	plt.colorbar()
	plt.title('R: N=%d, lppd=%4.3f' % (N, gpy_lppd))

	plt.subplot(2, len(Ns), 1 + len(Ns) + idx_N)
	plt.contourf(xs, ys, mu_ep, 30)
	plt.plot(X[:, 0], X[:, 1], 'r.', alpha=0.5)
	plt.colorbar()
	plt.title('U: N=%d, lppd=%4.3f' % (N, unimodal_lppd))
	plt.pause(1e-2)
	
	plt.show(block=False)

	fig.tight_layout()







