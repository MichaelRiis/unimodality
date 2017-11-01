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
from unimodal import UnimodalGP

from util import timeit
from util import plot_with_uncertainty, plot_with_uncertainty2
from scipy.stats import lognorm, norm

#############################################################################################
# Linear maps
#############################################################################################
class LinearMap(object):
	def __init__(self, a=0, b=1):
		self.a = a
		self.b = b

	def map(self, x):
		return (x - self.a)/(self.b - self.a)

	def inverse(self, y):
		return (self.b - self.a)*y + self.a



#############################################################################################
# Data
#############################################################################################
x = np.array([-0.86, -0.30, -0.05, 0.73])
n = np.array([5, 5, 5, 5])
y = np.array([0, 1, 3, 5])

# for plotting
A_original = np.linspace(-4, 8, 200)
# B_original = np.linspace(-10, 40, 200)
B_original = np.linspace(0, 50, 200)


Amap = LinearMap(A_original[0], A_original[-1])
Bmap = LinearMap(B_original[0], B_original[-1])

A = Amap.map(A_original)
B = Bmap.map(B_original)

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
dim = 1

# use same kernel for f
rbf, bias = GPy.kern.RBF(input_dim=1, lengthscale=0.1, variance=1.), GPy.kern.Bias(input_dim=1)

# priors for RBF
rbf.variance.set_prior(GPy.priors.HalfT(1,1), warning=False)
rbf.lengthscale.set_prior(GPy.priors.HalfT(1,1), warning=False)

# priors for bias
bias.variance.set_prior(GPy.priors.HalfT(1,1), warning=False)


@timeit
def fit_unimodal(X, Y):
	# kernel for g
	g_kernel_base = GPy.kern.RBF(input_dim=1, lengthscale=0.1, variance=1) 
	g_kernel_base.variance.set_prior(GPy.priors.LogGaussian(1., 0.5))
	g_kernel_base.lengthscale.set_prior(GPy.priors.LogGaussian(-1, 0.5))

	# likelihood
	lik = GPy.likelihoods.Gaussian(variance=1e-4)
	lik.variance.constrain_fixed()

	# create pseudo obserrvations
	M = 30
	Xd = np.linspace(0, 1, M)[:, None]
	m = -np.ones((2, M))

	Xq = [np.column_stack((Xd[[0, -1]], np.zeros((2, 1))))]
	Yq = [np.array([1, -1])[:, None]]


	# import ipdb; ipdb.set_trace()

	# fit model
	uni_gp = UnimodalGP(X=X, Y=Y, Xd=Xd, f_kernel_base=rbf.copy() + bias.copy(), g_kernel_base=g_kernel_base, likelihood=lik, m=m, Xq=Xq, Yq=Yq)
	uni_gp.optimize()

	return uni_gp

@timeit
def fit_regular(X, Y):
	reg_gp = GPy.models.GPRegression(X=X, Y=Y, kernel=rbf.copy() + bias.copy(), noise_var=1e-4)
	reg_gp.likelihood.variance.constrain_fixed()
	reg_gp.optimize()

	return reg_gp

@timeit
def fit_regular_gauss(X, Y):

	# get mean and variance of mean function from Laplace approximation
	mode, S = compute_laplace()

	# setup mean funciton
	mf = LogGaussMap(input_dim=dim, output_dim=1, mean=mode[:, None], cov=S)

	gauss_gp = GPy.models.GPRegression(X=X, Y=Y, kernel=rbf.copy() + bias.copy(), noise_var=1e-4, mean_function=mf)
	gauss_gp.likelihood.variance.constrain_fixed()
	gauss_gp.optimize()

	return gauss_gp

#############################################################################################
# Total variance
#############################################################################################
log_density_maps = {'mean': 	lambda mu, var=0: mu + 0.5*var,
					'mode': 	lambda mu, var=0: mu - var,
					'median': 	lambda mu, var=0: mu}



def compute_TV(model, log_marginal_posterior_b):
	p_true = np.exp(log_marginal_posterior_b)

	mu, var = model.predict(B[:, None])
	p_approx = np.exp(mu + 0.5*var).ravel()
	Zapprox = integrate1d(p_approx, B)
	p_approx = p_approx/Zapprox

	I = 0.5*np.abs(p_approx - p_true)

	return integrate1d(I.ravel(), B)


def compute_TV2(mu, var, log_marginal_posterior_b):
	p_true = np.exp(log_marginal_posterior_b)

	p_approx = np.exp(mu + 0.5*var).ravel()
	Zapprox = integrate1d(p_approx, B)
	p_approx = p_approx/Zapprox

	I = 0.5*np.abs(p_approx - p_true)

	return integrate1d(I.ravel(), B)


#############################################################################################
# KL-divergence
#############################################################################################


#############################################################################################
# Functions for computing prior, likelihood and posteriors
#############################################################################################
def log_likelihood(A,B,x,y,n):
	'''
	unnormalized log likelihood density for bioassay (assuming uniform prior)
	'''
	
	a = Amap.inverse(A)
	b = Bmap.inverse(B)

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

def compute_true_normalization():
	# define grid points for integration
	num_A, num_B = 300, 300
	A, B = np.linspace(0, 1, num_A), np.linspace(0, 1, num_B)

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

	return Ztrue

# pre-compute true normalization
Ztrue = compute_true_normalization()
logZtrue = np.log(Ztrue)


def log_marginal_posterior_b(b):
	
	# define grid points for integration
	num_A = 300
	A = np.linspace(0, 1, num_A)
	fB = np.stack([np.exp(log_posterior(a,b)) for a in A])

	return np.log(np.trapz(fB, A)) - logZtrue

def integrate1d(fX, X):
	return np.trapz(fX, X)



#############################################################################################
# Main
#############################################################################################
if __name__ == "__main__":

	snb.set(font_scale=0.8)


	# set seed
	seed = 0
	np.random.seed(seed)

	# compute true log marginal posterior
	lmp = np.stack([log_marginal_posterior_b(b) for b in B])

	# initial data point
	X = np.random.uniform(0, 1, size = (1, 1))
	Y = np.stack([log_marginal_posterior_b(xi) for xi in X])[:, None]
	X, Y = np.zeros((0, 1)), np.zeros((0, 1))
	max_itt = 10

	TVs = []

	fit_function = fit_unimodal
	# fit_function = fit_regular
	# fit_function = fit_regular_gauss

	fig = plt.figure()
	for itt in range(max_itt):

		t0 = time.time()

		# fit model
		if len(X) == 0 and fit_function is not fit_unimodal:

			if fit_function is fit_regular:
				mu = np.zeros((len(B), 1))
			else:
				
				# get mean and variance of mean function from Laplace approximation
				mode, S = compute_laplace()
				mf = LogGaussMap(input_dim=dim, output_dim=1, mean=mode[:, None], cov=S)
				mu = mf.f(B[:, None])
		
			var = (rbf.variance + bias.variance + 1e-4)*np.ones((len(B), 1))

		else:
			model = fit_function(X, Y)
			print(model)
			
			# predict
			mu, var = model.predict(B[:, None])
			

		
		density_mu = np.exp(mu + 0.5*var)
		density_var = (np.exp(var) - 1)*np.exp(2*mu + var)

		# compute TV
		TV = compute_TV2(mu, var, lmp)
		TVs.append(TV)

		# find next point
		maxval = np.max(density_var)
		idx = np.random.choice(np.where(density_var == maxval)[0], size=1)

		# idx = np.argmax(density_var)
		Xstar = np.atleast_2d(B[idx])
		Ystar = log_marginal_posterior_b(Xstar)

		t1 = time.time()
		print('Iteration %d done in %4.3fs' % (itt+1, t1-t0))

		# compute quantiles of log_density
		log_lower = [norm.ppf(0.05, loc=mui, scale=np.sqrt(vari)) for mui, vari in zip(mu, var)]
		log_upper = [norm.ppf(0.95, loc=mui, scale=np.sqrt(vari)) for mui, vari in zip(mu, var)]
		log_density_interval = np.column_stack((log_lower, log_upper))

		# compute quantiles of density
		density_interval = np.exp(np.column_stack((log_lower, log_upper)))

		# plot	
		plt.subplot2grid((4, max_itt), (0, itt))
		plt.plot(B, lmp)
		plt.plot(X, Y, 'k.')
		plot_with_uncertainty2(B, mu, lower=log_density_interval[:, 0], upper=log_density_interval[:, 1])
		plt.title('N = %d' % len(X))
		plt.grid(True)
		if itt == 0:
			plt.ylabel('Log density')

		plt.subplot2grid((4, max_itt), (1, itt))
		plt.plot(B, np.exp(lmp))
		plot_with_uncertainty2(B, density_mu, lower=density_interval[:, 0], upper=density_interval[:, 1])
		plt.title('TV = %4.3f' % TV)
		plt.grid(True)
		if itt == 0:
			plt.ylabel('Density')

		if fit_function is fit_unimodal:
			gmu, gvar = model.predict_g(B)
			plt.subplot2grid((4, max_itt), (2, itt))
			plot_with_uncertainty(B, gmu, gvar)
			plt.grid(True)
			plt.ylim((-4.5, 4.5))
			if itt == 0:
				plt.ylabel('g')

		plt.subplot2grid((4, max_itt), (3, itt))
		plt.plot(B, density_var)
		plt.plot(Xstar, density_var[idx], 'g.')
		plt.grid(True)
		if itt == 0:
			plt.ylabel('Acquisition')



		plt.pause(1e-3)
		plt.draw()

		X = np.row_stack((X, Xstar))
		Y = np.row_stack((Y, Ystar))


	fig.tight_layout()
	plt.show()