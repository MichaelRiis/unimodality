import numpy as np
import pylab as plt

from scipy.stats import norm, t

import GPy

from util import plot_with_uncertainty
import ep_unimodality_2d as ep
from importlib import reload
reload(ep)

phi = lambda x: norm.cdf(x)

# set seed
np.random.seed(0)

# Define test function
# f = lambda x, y: 0.1*(x**2 + y**2 + -1.2*x*y)
# f = lambda x, y: -t.logpdf(x-1, 1) -t.logpdf(y+2, 1)
f = lambda x, y: 0.1*((x-2)**2 + (y-2)**2 - 1.2*(x-2)*(y-2))


# For prediction and plotting
xs = np.linspace(-10, 10, 41)
ys = np.linspace(-10, 10, 41)
Xp, Yp = np.meshgrid(xs, ys)
XY = np.column_stack((Xp.ravel(), Yp.ravel()))
fp = f(Xp, Yp)

# sample points
sigma2 = 1e-1
N = 20
X = np.random.normal(0, 3.5, size = (N, 2))
y = f(X[:, 0], X[:, 1])[:, None] + np.random.normal(0, np.sqrt(sigma2), size=(N, 1))

# generate test set
Ntest = 1000
Xtest = np.random.normal(0, 3.5, size = (Ntest, 2))
ytest = f(Xtest[:, 0], Xtest[:, 1])[:, None] + np.random.normal(0, np.sqrt(sigma2), size=(Ntest, 1))

# GP hyperparameters
variance = 10
lengthscale = 4

####################################################################################################################################################3
# GPy
####################################################################################################################################################3

kernel = GPy.kern.RBF(input_dim=2, lengthscale=lengthscale, variance=variance)
if N > 0:
	gpy_model = GPy.models.GPRegression(X=X, Y=y, kernel=kernel, noise_var=sigma2)
	# model.optimize()
	mu_gpy, var_gpy = gpy_model.predict(XY)
	mu_gpy = mu_gpy.reshape((len(xs), len(ys)))
	var_gpy = var_gpy.reshape((len(xs), len(ys)))

	# evaluate
	lppd_gpy = np.mean(gpy_model.log_predictive_density(Xtest, ytest))
else:
	mu_gpy = np.zeros((len(xs), len(ys)))
	var_gpy = variance*np.ones((len(xs), len(ys)))

	log_npdf = lambda x, m, v: -0.5*np.log(2*np.pi*v) -(x-m)**2/(2*v)
	lppd_gpy = np.mean(log_npdf(ytest, mu_gpy.ravel(), var_gpy.ravel() + sigma2))

err_gpy = np.mean((fp.ravel() - mu_gpy.ravel())**2)/np.mean(fp.ravel()**2)

####################################################################################################################################################3
# Unimodal
####################################################################################################################################################3

# Fit Unimodal GP
M = 10
x1 = np.linspace(-10, 10, M)
x2 = np.linspace(-10, 10, M)
X1, X2 = np.meshgrid(x1, x2)
Xd = np.column_stack((X1.ravel(), X2.ravel()))

# fit
mu_f, Sigma_f, Sigma_full_f, g_posterior_list, Kf = ep.ep_unimodality(X, y, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, t2=Xd, verbose=10, nu2=1., c1=1, c2=lengthscale)

# predict
mu_ep, var_ep = ep.predict(mu_f, Sigma_full_f, X, [Xd, Xd], XY, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2)
mu_ep_g1, var_ep_g1 = ep.predict(g_posterior_list[0][0], g_posterior_list[0][2], Xd, [Xd, None], XY, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)
mu_ep_g2, var_ep_g2 = ep.predict(g_posterior_list[1][0], g_posterior_list[0][2], Xd, [None, Xd], XY, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2, f=False)


# evaluate
lppd_uni = ep.lppd(ytest, mu_f, Sigma_full_f, X, [Xd, Xd], Xtest, k1=np.sqrt(variance), k2=lengthscale, sigma2=sigma2)
err_uni = np.mean((fp.ravel() - mu_ep.ravel())**2)/np.mean(fp.ravel()**2)


####################################################################################################################################################3
# Evaluation
####################################################################################################################################################3
print(2*'\n')

names = ['GPy', 'Uni']
lppds = [lppd_gpy, lppd_uni]
nmses = [err_gpy, err_uni]

print(60*'-')
print('%10s\t%s\t\t%s' % ('Name', 'LPPD', 'NMSE'))
print(60*'-')

for name, lppd, nmse in zip(names, lppds, nmses):
	print('%10s\t%4.3f\t\t%4.3f' % (name, lppd, nmse))

####################################################################################################################################################3
# Plot
####################################################################################################################################################3

mu_ep = mu_ep.reshape((len(xs), len(ys)))
var_ep = var_ep.reshape((len(xs), len(ys)))
mu_ep_g1 = mu_ep_g1.reshape((len(xs), len(ys)))
mu_ep_g2 = mu_ep_g2.reshape((len(xs), len(ys)))

var_ep_g1 = var_ep_g1.reshape((len(xs), len(ys)))
var_ep_g2 = var_ep_g2.reshape((len(xs), len(ys)))


# for line plot
if N > 0:
	idx = np.argmin(np.sum(X**2, axis = 1))
	x0, y0 = X[idx, 0], X[idx, 1]
	idx2 = np.argmin((xs-x0)**2)
else:
	x0, y0 = 0, 0
	idx2 = np.argmin((xs-x0)**2)


# plot
fig = plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.contourf(xs, ys, f(Xp, Yp), 30)
plt.axhline(y0, color='k', linestyle='--')
plt.plot(X[:, 0], X[:, 1], 'r.')
plt.colorbar()
plt.title('True functions and observations')

plt.subplot(2, 3, 2)
plt.contourf(xs, ys, mu_gpy, 30)
plt.plot(X[:, 0], X[:, 1], 'r.')
plt.colorbar()
plt.title('Regular GP')

plt.subplot(2, 3, 3)
plt.contourf(xs, ys, mu_ep, 30)
plt.plot(X[:, 0], X[:, 1], 'r.')
plt.colorbar()
plt.title('Unimodal GP')

if M > 0:

	plt.subplot(2, 3, 5)
	mu_g = g_posterior_list[0][0][:(len(x1)*len(x2))].reshape((len(x1), len(x2)))
	plt.contourf(xs, xs, phi(mu_ep_g1/np.sqrt(1+var_ep_g1)), 30)
	plt.plot(Xd[:, 0], Xd[:, 1], 'k.', markersize = 2, alpha=0.75)
	plt.colorbar()
	plt.title('Posterior mean deriv1')

	plt.subplot(2, 3, 6)
	# mu_g = g_posterior_list[1][0][:(len(x1)*len(x2))].reshape((len(x1), len(x2)))
	plt.contourf(xs, ys, phi(mu_ep_g2/np.sqrt(1+var_ep_g2)), 30)
	plt.plot(Xd[:, 0], Xd[:, 1], 'k.', markersize = 2, alpha=0.75)
	plt.colorbar()
	plt.title('Posterior mean deriv2')

fig.tight_layout()



if M > 0:
	plt.figure()
	plt.subplot(1, 2, 1)
	mu_g = g_posterior_list[0][0]
	mu_g = mu_g[(M**2):2*(M**2)]
	mu_g = mu_g.reshape((M, M))

	plt.contourf(x1, x2, mu_g, 30)
	plt.colorbar()

	plt.subplot(1, 2, 2)
	mu_g = g_posterior_list[1][0]
	mu_g = mu_g[(M**2):2*(M**2)]
	mu_g = mu_g.reshape((M, M))

	plt.contourf(x1, x2, mu_g, 30)
	plt.colorbar()


# fit line
plt.figure()
plt.plot(xs, f(xs, y0), label = 'True function', color='g')
# plt.plot(x0, f(x0, y0), 'k.')
plot_with_uncertainty(xs,  mu_ep[idx2, :], ystd=np.sqrt(var_ep[idx2, :]), color='r', label='Unimodal', linestyle='--')
plot_with_uncertainty(xs,  mu_gpy[idx2, :], ystd=np.sqrt(var_gpy[idx2, :]), color='b', label='Vanilla')
plt.legend()
plt.grid(True)
plt.show(block=False)


plt.show()





