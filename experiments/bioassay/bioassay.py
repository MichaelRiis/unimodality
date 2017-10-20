import autograd.numpy as np
import pylab as plt
import seaborn as snb
import time
import argparse
import GPy
from GPy.core.parameterization import Param
from GPy.util import choleskies


from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
from autograd import value_and_grad, grad, hessian

import sys
sys.path.append('../../code')
import unimodal 

from util import timeit


#############################################################################################
# Data
#############################################################################################
x = np.array([-0.86, -0.30, -0.05, 0.73])
n = np.array([5, 5, 5, 5])
y = np.array([0, 1, 3, 5])

# for plotting
A = np.linspace(-4, 8, 200)
B = np.linspace(-10, 40, 200)

#############################################################################################
# Functions for using log gauss density as mean function
#############################################################################################
def log_mvn_chol(x, mu, L):

    D = x.shape[1]

    logdet = 2*np.sum(np.log(np.diag(L)))
    det = np.exp(logdet)
    v = x - mu
    b = np.linalg.solve(L, v.T).T

    return -0.5*D*np.log(2*np.pi) - 0.5*logdet -0.5*np.sum(b**2, axis=1, keepdims=True)

class LogGaussMap(GPy.core.Mapping):

    def __init__(self, input_dim, output_dim, name='gaussmap', mean=None, cov=None):
        super(LogGaussMap, self).__init__(input_dim=input_dim, output_dim=output_dim, name=name)
        
        self.mean = Param('mean', np.zeros((input_dim, 1)))
        if not mean is None:
            self.mean = mean
            
        # initial chol
        if cov is None:
            C = np.identity(input_dim)
        else:
            C = np.linalg.cholesky(cov)

        chols = choleskies.triang_to_flat(np.tile(C[None,:, :], (1,1,1)))
        self.chols = Param('chols', chols)

    def f(self, X):
        L = choleskies.flat_to_triang(self.chols)[0]
        log_pdf = log_mvn_chol(X, self.mean.ravel(), L)
        return log_pdf
        

    def update_gradients(self, dL_dF, X):
        pass

    def gradients_X(self, dL_dF, X):
        return None



def compute_laplace():
    obj = lambda x: -log_posterior(x[0], x[1])

    res = minimize(value_and_grad(obj), np.array((0.3, 10.)), jac=True)
    mode = res.x
    hess = hessian(obj)(mode)
    S = 2*np.linalg.inv(hess)

    print('Found mode\t(a,b) = %r' % mode)
    print('Found covariance\t (a,b) = %r' % S)

    return mode, S



#############################################################################################
# Function for fitting and predicting
#############################################################################################

# use same kernel for f
rbf = GPy.kern.RBF(input_dim=2, lengthscale=1., variance=1.)
bias = GPy.kern.Bias(input_dim=2)

# add priors for rbf
rbf.variance.unconstrain()
rbf.variance.set_prior(GPy.priors.HalfT(1,1), warning=False)

# add priors for bias
bias.variance.unconstrain()
bias.variance.set_prior(GPy.priors.HalfT(1,1), warning=False)

@timeit
def fit_unimodal(X, Y):
	g_kernel_base = GPy.kern.RBF(input_dim=2, lengthscale=10, variance=0.1)

	# set priors
	g_kernel_base.variance.unconstrain()
	g_kernel_base.variance.set_prior(GPy.priors.LogGaussian(1., 0.5))

	g_kernel_base.lengthscale.unconstrain()
	g_kernel_base.lengthscale.set_prior(GPy.priors.LogGaussian(-1, 0.1))

	lik = GPy.likelihoods.Gaussian(variance=1e-3)
	lik.variance.constrain_fixed()

	# create pseudo observations
	M = 8
	x1 = np.linspace(A[0], A[-1], M)
	x2 = np.linspace(B[0], B[-1], M)
	X1, X2 = np.meshgrid(x1, x2)
	Xd = np.column_stack((X1.ravel(), X2.ravel()))

	uni_gp = unimodal.UnimodalGP(X=X, Y=Y, Xd=Xd, f_kernel_base=rbf.copy() + bias.copy(), g_kernel_base=g_kernel_base, likelihood=lik)
	uni_gp.optimize()

	return uni_gp

@timeit
def fit_regular(X, Y):
	reg_gp = GPy.models.GPRegression(X=X, Y=Y, kernel=rbf.copy() + bias.copy(), noise_var=1e-3)
	reg_gp.likelihood.variance.constrain_fixed()
	reg_gp.optimize()

	return reg_gp

@timeit
def fit_regular_gauss(X, Y):

	# get mean and variance of mean function from Laplace approximation
	mode, S = compute_laplace()

	# setup mean funciton
	mf = LogGaussMap(input_dim=2, output_dim=1, mean=mode[:, None], cov=S)

	gauss_gp = GPy.models.GPRegression(X=X, Y=Y, kernel=rbf.copy() + bias.copy(), noise_var=1e-3, mean_function=mf)
	gauss_gp.likelihood.variance.constrain_fixed()
	gauss_gp.optimize()

	return gauss_gp


@timeit
def predict_grid(model, A, B):

	AA, BB = np.meshgrid(A, B)
	AB = np.column_stack((AA.ravel(), BB.ravel()))

	mu, var = model.predict(AB)
	mu = -mu.reshape((len(A), len(B)))
	var = var.reshape((len(A), len(B)))

	return mu, var

#############################################################################################
# KL-divergence
#############################################################################################
log_density_maps = {'mean': 	lambda mu, var: mu + 0.5*var,
					'mode': 	lambda mu, var: mu - var,
					'median': 	lambda mu, var: mu}

@timeit
def compute_KL(model, log_map='mean'):

	# get map
	log_density_map = log_density_maps[log_map]
	
	# define grid points for integration
	num_A, num_B = 300, 300
	A, B = np.linspace(-4, 8, num_A), np.linspace(-10, 40, num_B)

	# predict approximate density for unimodal
	mu, var = predict_grid(model, A, B)

	# evaluate true posterior
	log_true = evaluate_log_posterior_grid(A, B)

	def integrate(W):
		""" Helper function """
		# do a 1-D integral over every row
		I = np.zeros( num_B )
		for i in range(num_B):
		    I[i] = np.trapz( W[i,:], A )
		# then an integral over the result
		return np.trapz( I, B )

	# compute normalizations
	Ztrue = integrate(np.exp(log_true))

	# compute log approximation and it's normalization
	log_approx = log_density_map(mu, var)
	log_approx -= np.max(log_approx)
	Zapprox = integrate(np.exp(log_approx))

	# compute KL
	R = integrate(np.exp(log_true)*(log_true - log_approx))
	KL = R/Ztrue - np.log(Ztrue) + np.log(Zapprox)

	return KL


def compute_gauss_KL(mu, cov):


	# define grid points for integration
	num_A, num_B = 300, 300
	A, B = np.linspace(-4, 8, num_A), np.linspace(-10, 40, num_B)

	# evaluate true posterior
	log_true = evaluate_log_posterior_grid(A, B)

	def integrate(W):
		""" Helper function """
		# do a 1-D integral over every row
		I = np.zeros( num_B )
		for i in range(num_B):
		    I[i] = np.trapz( W[i,:], A )
		# then an integral over the result
		return np.trapz( I, B )

	# compute normalizations
	Ztrue = integrate(np.exp(log_true))

	# compute log approximation log(E[exp(f)]) and it's normalization
	# log_approx = mu + 0.5*var
	L = np.linalg.cholesky(cov)
	AA, BB = np.meshgrid(A, B)
	AB = np.column_stack((AA.ravel(), BB.ravel()))
	log_approx = log_mvn_chol(AB, mu, L).reshape((len(A), len(B)))
	log_approx -= np.max(log_approx)
 
	Zapprox = integrate(np.exp(log_approx))

	# compute KL
	R = integrate(np.exp(log_true)*(log_true - log_approx))
	KL = R/Ztrue - np.log(Ztrue) + np.log(Zapprox)

	return KL


#############################################################################################
# Functions for computing prior, likelihood and posteriors
#############################################################################################
def log_likelihood(a,b,x,y,n):
	'''
	unnormalized log likelihood density for bioassay (assuming uniform prior)
	'''
	
	# these help using chain rule in derivation
	t = a + b*x
	et = np.exp(t)

	# negative log posterior (error function to be minimized)
	lp = np.sum( y*t  -n*np.log1p(et) )

	return lp 

def log_prior(a, b):
	return 0


def log_posterior(a, b):
	return log_prior(a, b) + log_likelihood(a, b, x, y, n)

@timeit
def evaluate_log_posterior_grid(A, B):
	
	# evaluate prior & likelihood
	log_posterior_grid = np.zeros((len(A), len(B)))

	for i, a in enumerate(A):
		for j, b in enumerate(B):
			log_posterior_grid[i,j] = log_posterior(a, b)

	return log_posterior_grid


#############################################################################################
# Plotting
#############################################################################################


def plot_contour(Z, title='', levels=None):

	if levels is None:
		plt.contourf(A, B, Z, 30);	
	else:
		plt.contourf(A, B, Z, levels);	
	plt.title(title)
	plt.xlabel('Alpha')
	plt.ylabel('Beta')
	plt.colorbar()


#############################################################################################
# Main
#############################################################################################
if __name__ == "__main__":
	
	# parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--N', type=int, help='Number of points', default = 30)
	parser.add_argument('--seed', type=int, help='seed', default=1000)
	args = parser.parse_args()
	N, seed = args.N, args.seed

	# set seed
	np.random.seed(seed)

	# computer posterior in grid
	log_posterior_grid = evaluate_log_posterior_grid(A, B)

	# sample initial points and evaluate posterior at these poins
	N = 50
	X = np.random.uniform(size = (N, 2))*np.array([12, 50]) + np.array([-4, -10])
	Y = np.stack([-log_posterior(a,b) for (a,b) in X])[:, None]


	# fit regular GP
	reg_gp  = fit_regular(X, Y)
	reg_mu, reg_var = predict_grid(reg_gp, A, B)

	# fit unimodal GP
	uni_gp = fit_unimodal(X, Y)
	uni_mu, uni_var = predict_grid(uni_gp, A, B)

	# compute KL divergence
	uni_KL = compute_KL(uni_gp)
	reg_KL = compute_KL(reg_gp)

	print(100*'-')
	print('- KL divergence uisng N = %d' % N)
	print(100*'-')
	print('Uni KL: %5.4e' % uni_KL)
	print('Reg KL: %5.4e\n\n' % reg_KL)


	# how to map from log_density to density
	# map_to_density = lambda mu, var: np.exp(mu + 0.5*var)
	map_to_density = lambda mu, var: np.exp(mu)
	density_levels = None#np.linspace(0, (0.01), 30)


	# plot true posterior
	fig = plt.figure(figsize = (16, 10))
	plt.subplot(2, 3, 1)
	plot_contour(np.exp(log_posterior_grid), 'True posterior', levels=density_levels)
	plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

	# plot regular gp approximation
	plt.subplot(2, 3, 2)
	plot_contour(map_to_density(reg_mu, reg_var), 'GP approximation', levels=density_levels)
	plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

	# plot regular gp approximation error
	plt.subplot(2, 3, 3)
	plot_contour(np.exp(log_posterior_grid) - map_to_density(reg_mu, reg_var), 'GP approximation error')
	plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

	# plot unimodal gp
	plt.subplot(2, 3, 5)
	plot_contour(map_to_density(uni_mu, uni_var), 'Uni approximation', levels=density_levels)
	plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

	# plot unimodal gp approximation error
	plt.subplot(2, 3, 6)
	plot_contour(np.exp(log_posterior_grid) - map_to_density(uni_mu, uni_var), 'Uni approximation error')
	plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

	fig.tight_layout()

	plt.show()