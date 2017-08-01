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



fs = [lambda x: -0.4*(norm.logpdf(x-2, 0, 1)) - 5,
	  lambda x: -4*(tdist.logpdf(x+2, 1) + 2.8),
	  lambda x: 0.66*(-6.47022e-01*x**0 + -3.70935e-14*x**1 + -2.60029e-02*x**2 + 6.82401e-15*x**3 + 4.40170e-02*x**4 + -2.76093e-16*x**5 + -1.57267e-03*x**6 + 2.97028e-18*x**7 + 1.60553e-05*x**8),	
	  lambda x: 0.3*(x**2 - 25)*np.sin(x+0.5*np.pi),
      lambda x: 5*np.sin(x+0.5*np.pi)]


sigma2 = 5.

k1, k2 = 5., 2.

N = 20
M = 20
t = np.linspace(-5, 5, N)[:, None]
t2 = np.linspace(-7, 7, M)[:, None]
ys = [f(t) + np.random.normal(0, np.sqrt(sigma2), size=(N, 1)) for f in fs]
t_pred = np.linspace(-7, 7, 101)[:, None]


fig = plt.figure(figsize = (16, 9))
for idx in range(len(ys)):
	f, y = fs[idx](t_pred), ys[idx]

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
	plt.subplot(3, len(ys), 1 + idx)
	plt.plot(t_pred, f, 'g-', linewidth = 3., label='True function')
	plot_with_uncertainty(x=t_pred, y=mu_f_pred, ystd=np.sqrt(sigma_f_pred), label='Unimodal-GP')
	plot_with_uncertainty(x=t_pred, y=mu_gpy, ystd=np.sqrt(sigma_gpy), color='b', fill=False, linestyle='--', label = 'Vanilla-GP')
	plt.plot(t, y, 'k.', markersize=8, label='Data')
	plt.plot(t2, -10*np.ones_like(t2), 'rx', markersize = 6)
	plt.ylim((-12, 12))
	plt.grid(True)

	plt.subplot(3, len(ys),  1 + len(ys) + idx)
	pz = phi(mu_g_pred/np.sqrt(1 + sigma_g_pred))
	pz_mean, pz_var = EP.sample_z_probabilities(mu_g, Sigma_full_g, t2, t2, t_pred, k1, k2)
	plot_with_uncertainty(x=t_pred, y=pz_mean, ystd=np.sqrt(pz_var))
	plt.grid(True)

	plt.subplot(3, len(ys),  1 + 2*len(ys) + idx)
	plot_with_uncertainty(x=t_pred, y=mu_g_pred, ystd=np.sqrt(sigma_g_pred))
	plt.grid(True)
	plt.ylim((-12, 12))
	fig.tight_layout()
	plt.pause(1e-3)
plt.show(block = False)