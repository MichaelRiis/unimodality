import numpy as np
import pylab as plt
import seaborn as snb
import argparse
import GPy

from autograd import value_and_grad, grad, hessian
from scipy.optimize import minimize



import sys
sys.path.append('../../code')
from util import plot_with_uncertainty
import bioassay

from unimodal import UnimodalGP


###################################################################################3
# Generate data initial data
###################################################################################3
noise_var = 0.001

def sample(bs, a=0.5, noise=0):
    y = np.stack([-bioassay.log_posterior(a, bi)+np.random.normal(0,np.sqrt(noise)) for bi in bs])[:, None]
    return (y-9)/3

np.random.seed(1000)

# sweep range
Ns = np.arange(0, 10+1)

# for plotting
Bs = np.linspace(0, 1, 101)[:, None]
f = sample(Bs)


###################################################################################3
# Fit and plot for each N starting with 1
###################################################################################3

X = 0.5*np.ones((1,1))
Y = sample(X, noise=noise_var)

fig = plt.figure()
for idx_N, N in enumerate(Ns):

    # use same kernel for f
    rbf = GPy.kern.RBF(input_dim=1, lengthscale=0.1, variance=1.)
    bias = GPy.kern.Bias(input_dim=1)

    # add priors for rbf
    rbf.variance.unconstrain()
    rbf.variance.set_prior(GPy.priors.HalfT(1,1), warning=False)
    rbf.lengthscale.set_prior(GPy.priors.LogGaussian(1, 0.5), warning=False)
    rbf.lengthscale.constrain_fixed()
    rbf.variance.constrain_fixed()

    # add priors for bias
    bias.variance.unconstrain()
    bias.variance.set_prior(GPy.priors.HalfT(1,1), warning=False)
    bias.variance.constrain_fixed()

    g_kernel_base = GPy.kern.RBF(input_dim=1, lengthscale=0.1, variance=5) 

    # set priors
    g_kernel_base.variance.set_prior(GPy.priors.LogGaussian(3., 0.5))
    g_kernel_base.lengthscale.set_prior(GPy.priors.LogGaussian(-3, 0.5))
    # g_kernel_base.lengthscale.constrain_fixed()
    g_kernel_base.variance.constrain_fixed()

    # likelihood
    lik = GPy.likelihoods.Gaussian(variance=noise_var)
    lik.variance.constrain_fixed()

    # create pseudo obserrvations
    M = 30
    Xd = np.linspace(0, 1, M)[:, None]

    # fit model
    model = UnimodalGP(X=X, Y=Y, Xd=Xd, f_kernel_base=rbf.copy() + bias.copy(), g_kernel_base=g_kernel_base, likelihood=lik)
    # model = GPy.core.GP(X=X, Y=Y, kernel=rbf.copy() + bias.copy(), likelihood=lik)
    model.optimize()
    print(model)

    # predict
    mu, var = model.predict(Bs)
    density_mu = np.exp(-mu + 0.5*var)
    density_var = (np.exp(var) - 1)*np.exp(-2*mu + var)

    # find next point
    idx = np.argmax(density_var)
    Xstar = Bs[idx][:, None]
    Ystar = sample(Xstar, noise=noise_var)

    # plot
    plt.subplot2grid((3, len(Ns)), (0, idx_N))
    plt.plot(Bs, f)
    plt.plot(X, Y, 'k.')
    plot_with_uncertainty(Bs, mu, var)
    plt.title('Log, N = %d' % N)
    plt.grid(True)

    plt.subplot2grid((3, len(Ns)), (1, idx_N))
    plt.plot(Bs, np.exp(-f))
    plot_with_uncertainty(Bs, density_mu, density_var, color='r')
    plt.plot(X, np.exp(-Y), 'k.')
    plt.title('Density, N = %d' % N)
    plt.grid(True)


    plt.subplot2grid((3, len(Ns)), (2, idx_N))
    plt.plot(Bs, density_var)
    plt.plot(Xstar, density_var[idx], 'g.')
    plt.grid(True)
    plt.title('Acqusition')

    plt.pause(1e-3)
    plt.draw()

    # add new point
    X = np.row_stack((X, Xstar))
    Y = np.row_stack((Y, Ystar))

fig.tight_layout()
    
plt.show()
