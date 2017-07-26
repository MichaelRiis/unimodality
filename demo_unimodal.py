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


fs = [lambda x: -norm.logpdf(x-1, 0, 1) - 7,
	  lambda x: -4*(tdist.logpdf(x+2, 1) + 2.8),
	  lambda x: 0.3*(x**2 - 25)*np.sin(x+0.5*np.pi),
      lambda x: 5*np.sin(x+0.5*np.pi)]

sigma2 = 10.

k1, k2 = 5., 2.

N = 10
M = 50
t = np.linspace(-5, 5, N)[:, None]
t2 = np.linspace(-7, 7, M)[:, None]
ys = [f(t) + np.random.normal(0, np.sqrt(sigma2), size=(N, 1)) for f in fs]

t_pred = np.linspace(-7, 7, 101)[:, None]


plt.figure(figsize = (16, 9))
for idx in range(len(ys)):
	f, y = fs[idx](t_pred), ys[idx]

	# Fit model
	mu_f, Sigma_f, Sigma_full_f, mu_g, Sigma_g, Sigma_full_g = EP.ep_unimodality(t, y, k1, k2, sigma2, t2=t2)

	# Predict
	mu_f_pred, sigma_f_pred = EP.predict(mu_f, Sigma_full_f, t, t2, t_pred, k1, k2)
	mu_g_pred, sigma_g_pred = EP.predict(mu_g, Sigma_full_g, t2, t2, t_pred, k1, k2)


	model = GPy.models.GPRegression(X=t, Y=y, kernel=GPy.kern.RBF(input_dim=1, lengthscale=k2, variance=k1**2), noise_var=sigma2)
	mu_gpy, Sigma_gpy = model.predict(Xnew=t_pred, full_cov=True, include_likelihood=False)
	sigma_gpy = np.diag(Sigma_gpy)[:, None]

	plt.subplot(3, len(ys), 1 + idx)
	plt.plot(t_pred, f, 'g-', linewidth = 3., label='True function')
	plot_with_uncertainty(x=t_pred, y=mu_f_pred, ystd=np.sqrt(sigma_f_pred), label='Unimodal-GP')
	plot_with_uncertainty(x=t_pred, y=mu_gpy, ystd=np.sqrt(sigma_gpy), color='b', fill=False, linestyle='--', label = 'Vanilla-GP')
	plt.plot(t, y, 'k.', markersize=10, label='Data')
	plt.grid(True)

	plt.subplot(3, len(ys),  1 + len(ys) + idx)
	pz = phi(mu_g_pred/np.sqrt(1 + sigma_g_pred))
	pz_mean, pz_var = EP.sample_z_probabilities(mu_g, Sigma_full_g, t2, t2, t_pred, k1, k2)
	plot_with_uncertainty(x=t_pred, y=pz_mean, ystd=np.sqrt(pz_var))
	plt.grid(True)

	plt.subplot(3, len(ys),  1 + 2*len(ys) + idx)
	plot_with_uncertainty(x=t_pred, y=mu_g_pred, ystd=np.sqrt(sigma_g_pred))
	plt.grid(True)

	plt.pause(1e-3)
	plt.show(block = False)