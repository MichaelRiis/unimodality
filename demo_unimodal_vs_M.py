import numpy as np
import pylab as plt

from importlib import reload

from scipy.stats import norm, t as tdist

import GPy

import ep_unimodality as EP
reload(EP)
from ep_unimodality import phi
from util import plot_with_uncertainty

import seaborn as snb

np.random.seed(0)




fnc = lambda x: -0.4*(norm.logpdf(x-2, 0, 1)) - 5

sigma2 = 5.

k1, k2 = 5., 2.

N = 20
Ms = [0, 2, 4, 6, 8, 10, 20, 50]

t = np.linspace(-5, 5, N)[:, None]
y = fnc(t) + np.random.normal(0, np.sqrt(sigma2), size=(N, 1))
	

fig = plt.figure(figsize = (20, 9))
for idx in range(len(Ms)):


	M = Ms[idx]

	t2 = np.linspace(-7, 7, M)[:, None]
	t_pred = np.linspace(-7, 7, 101)[:, None]
	f = fnc(t_pred)


	# Fit model and predict
	mu_f, Sigma_f, Sigma_full_f, mu_g, Sigma_g, Sigma_full_g, logZ = EP.ep_unimodality(t, y, k1, k2, sigma2, t2=t2)
	mu_f_pred, sigma_f_pred = EP.predict(mu_f, Sigma_full_f, t, t2, t_pred, k1, k2)
	mu_g_pred, sigma_g_pred = EP.predict(mu_g, Sigma_full_g, t2, t2, t_pred, k1, k2)

	# Fit vanilla
	model = GPy.models.GPRegression(X=t, Y=y, kernel=GPy.kern.RBF(input_dim=1, lengthscale=k2, variance=k1**2), noise_var=sigma2)
	mu_gpy, Sigma_gpy = model.predict(Xnew=t_pred, full_cov=True, include_likelihood=False)
	sigma_gpy = np.diag(Sigma_gpy)[:, None]

	# print
	print('Log Z (vanilla): %5.4f' % (-model.objective_function()))
	print('Log Z (unimoda): %5.4f' % (logZ))
	print('\n')

	# Plot
	plt.subplot(3, len(Ms), 1 + idx)
	plt.plot(t_pred, f, 'g-', linewidth = 3., label='True function')
	plot_with_uncertainty(x=t_pred, y=mu_f_pred, ystd=np.sqrt(sigma_f_pred), label='Unimodal-GP')
	plot_with_uncertainty(x=t_pred, y=mu_gpy, ystd=np.sqrt(sigma_gpy), color='b', fill=False, linestyle='--', label = 'Vanilla-GP')
	plt.plot(t, y, 'k.', markersize=8, label='Data')
	plt.plot(t2, -10*np.ones_like(t2), 'rx', markersize = 6)
	plt.title('M = %d, logZ = %3.2f' % (M, logZ))
	plt.grid(True)
	plt.ylim((-11, 11))

	plt.subplot(3, len(Ms),  1 + len(Ms) + idx)
	pz = phi(mu_g_pred/np.sqrt(1 + sigma_g_pred))
	pz_mean, pz_var = EP.sample_z_probabilities(mu_g, Sigma_full_g, t2, t2, t_pred, k1, k2)
	plot_with_uncertainty(x=t_pred, y=pz_mean, ystd=np.sqrt(pz_var))
	plt.grid(True)

	plt.subplot(3, len(Ms),  1 + 2*len(Ms) + idx)
	plot_with_uncertainty(x=t_pred, y=mu_g_pred, ystd=np.sqrt(sigma_g_pred))
	plt.grid(True)
	plt.ylim((-12, 12))
	fig.tight_layout()

	plt.pause(1e-3)
plt.show(block = False)








# xs = np.array([-7.5, -6, -5, -4, -3, -2, -1.5, -1, 0, 1, 1.5, 2, 3, 4, 5, 6, 7.5])
# ys = np.array([18, 9, 8,  5,  1,  0,  0, 0, -3, 0, 0, 0, 1, 5, 8, 9, 18])
# P = 8

# X = np.zeros((len(xs), P + 1))
# for n in range(P + 1):
# 	X[:, n] = xs**n


# w = np.linalg.solve(np.dot(X.T, X) + 1*np.identity(P+1), np.dot(X.T, ys))

# xp = np.linspace(-7.5, 7.5, 101)
# X = np.zeros((len(xp), P + 1))
# for n in range(P + 1):
# 	X[:, n] = xp**n

# yhat = np.dot(X, w)
# plt.plot(xp, yhat)
# plt.plot(xs, ys, 'k.')
# plt.show()


#  s = ' + '.join(['%6.5e*x**%d' % (w, n) for n, w in zip(range(P+1), w)])

