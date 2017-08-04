import numpy as np
import pylab as plt

from importlib import reload

from scipy.stats import norm, t as tdist

import GPy

import ep_unimodality as EP
reload(EP)
from ep_unimodality import phi
from util import plot_with_uncertainty
from moment_functions import compute_moments_softinformation, compute_moments_strict


import seaborn as snb

np.random.seed(0)



fs = [lambda x: -0.4*(norm.logpdf(x-2, 0, 1)) - 5,
	  lambda x: -4*(tdist.logpdf(x+2, 1) + 2.8),
	  lambda x: 0.66*(-6.47022e-01*x**0 + -3.70935e-14*x**1 + -2.60029e-02*x**2 + 6.82401e-15*x**3 + 4.40170e-02*x**4 + -2.76093e-16*x**5 + -1.57267e-03*x**6 + 2.97028e-18*x**7 + 1.60553e-05*x**8),	
	  lambda x: 0.3*(x**2 - 25)*np.sin(x+0.5*np.pi),
      lambda x: 5*np.sin(x+0.5*np.pi)]


sigma2 = 5.

k1, k2 = 5., 2.
c1, c2 = 0.5, 2.
N = 20
M = 60
t = np.linspace(-5, 5, N)[:, None]
t2 = np.linspace(-7, 7, M)[:, None]
ys = [f(t) + np.random.normal(0, np.sqrt(sigma2), size=(N, 1)) for f in fs]
t_pred = np.linspace(-7, 7, 101)[:, None]


fig = plt.figure(figsize = (16, 9))
for idx in range(len(ys)):
	f, y = fs[idx](t_pred), ys[idx]

	# Fit vanilla
	model = GPy.models.GPRegression(X=t, Y=y, kernel=GPy.kern.RBF(input_dim=1, lengthscale=k2, variance=k1**2), noise_var=sigma2)
	mu_gpy, Sigma_gpy = model.predict(Xnew=t_pred, full_cov=True, include_likelihood=False)
	sigma_gpy = np.diag(Sigma_gpy)[:, None]
	print('\nLog Z (vanilla): %5.4f' % (-model.objective_function()))

	# Fit model and predict with strict moments
	mu_f_hard, Sigma_f_hard, Sigma_full_f_hard, mu_g_hard, Sigma_g_hard, Sigma_full_g_hard, logZ_hard = EP.ep_unimodality(t, y, k1, k2, sigma2, t2=t2, moment_function=compute_moments_strict, c1=c1, c2=c2)
	mu_f_pred_hard, sigma_f_pred_hard = EP.predict(mu_f_hard, Sigma_full_f_hard, t, t2, t_pred, k1, k2)
	mu_g_pred_hard, sigma_g_pred_hard = EP.predict(mu_g_hard, Sigma_full_g_hard, t2, t2, t_pred, k1, k2)
	print('Log Z (unimoda-hard): %5.4f' % (logZ_hard))

	# Fit model and predict with soft moments
	mu_f_soft, Sigma_f_soft, Sigma_full_f_soft, mu_g_soft, Sigma_g_soft, Sigma_full_g_soft, logZ_soft = EP.ep_unimodality(t, y, k1, k2, sigma2, t2=t2, moment_function=compute_moments_softinformation, c1=c1, c2=c2)
	mu_f_pred_soft, sigma_f_pred_soft = EP.predict(mu_f_soft, Sigma_full_f_soft, t, t2, t_pred, k1, k2)
	mu_g_pred_soft, sigma_g_pred_soft = EP.predict(mu_g_soft, Sigma_full_g_soft, t2, t2, t_pred, k1, k2)
	print('Log Z (unimoda-soft): %5.4f' % (logZ_soft))
	

	# Plot
	plt.subplot(2, len(ys), 1 + idx)
	plt.plot(t_pred, f, 'g-', linewidth = 3.)
	plot_with_uncertainty(x=t_pred, y=mu_f_pred_hard, ystd=np.sqrt(sigma_f_pred_hard), color='r', label='Unimodal-hard')
	plot_with_uncertainty(x=t_pred, y=mu_f_pred_soft, ystd=np.sqrt(sigma_f_pred_soft), color='y', label='Unimodal-soft', linestyle='--')
	plot_with_uncertainty(x=t_pred, y=mu_gpy, ystd=np.sqrt(sigma_gpy), color='b', fill=False, linestyle='--')
	plt.plot(t, y, 'k.', markersize=8)
	plt.plot(t2, -10*np.ones_like(t2), 'rx', markersize = 6)
	plt.ylim((-12, 12))
	plt.grid(True)
	plt.legend()

	plt.subplot(2, len(ys),  1 + len(ys) + idx)
	pz_hard = 2*phi(mu_g_pred_hard/np.sqrt(1 + sigma_g_pred_hard)) - 1
	plot_with_uncertainty(x=t_pred, y=pz_hard, label = 'hard', color='r')

	pz = 2*phi(mu_g_pred_soft/np.sqrt(1 + sigma_g_pred_soft)) - 1
	plot_with_uncertainty(x=t_pred, y=pz, label = 'soft', color='y', linestyle = '--')
	plt.grid(True)

	# plt.subplot(3, len(ys),  1 + 2*len(ys) + idx)
	
	# plt.grid(True)

	fig.tight_layout()
	plt.pause(1e-3)
plt.show(block = False)

