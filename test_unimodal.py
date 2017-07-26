import numpy as np
import pylab as plt

from importlib import reload

from scipy.stats import norm, t as tdist

import GPy


import ep_unimodality as EP
reload(EP)
from ep_unimodality import phi

import seaborn as snb

np.random.seed(0)

def plot_with_uncertainty(x, y, ystd=None, color='r', linestyle='-', fill=True, label=''):
    plt.plot(x, y, color=color, linestyle=linestyle, label=label)
    if not ystd is None:
        lower, upper = y - np.sqrt(ystd), y + np.sqrt(ystd)
        plt.plot(x, lower, color=color, alpha=0.5, linestyle=linestyle)
        plt.plot(x, upper, color=color, alpha=0.5, linestyle=linestyle)
        
        if fill:
            plt.fill_between(x.ravel(), lower, upper, color=color, alpha=0.25, linestyle=linestyle)


fs = [lambda x: -norm.logpdf(x-1, 0, 1) - 7, lambda x: -4*(tdist.logpdf(x+2, 1) + 2.8), lambda x: 0.3*(x**2 - 25)*np.sin(x+0.5*np.pi), lambda x: 5*np.sin(x+0.5*np.pi)]

sigma2 = 10.

k1, k2 = 5., 2.

N = 20
M = 20
t = np.linspace(-5, 5, N)[:, None]
t2 = np.linspace(-5, 5, M)[:, None]
ys = [f(t) + np.random.normal(0, np.sqrt(sigma2), size=(N, 1)) for f in fs]

# plt.figure(figsize = (25, 5))
# for idx, (y, f) in enumerate(zip(ys, fs)):
#     plt.subplot(1, len(fs), 1 + idx)
#     plt.plot(t, f(t))
#     plt.plot(t, y, 'k.')


plt.figure(figsize = (16, 9))
for idx in range(len(ys)):
    f, y = fs[idx](t), ys[idx]

    mu_f, Sigma_f, mu_g, Sigma_g = EP.ep_unimodality(t, y, k1, k2, sigma2, t2=t2)
    model = GPy.models.GPRegression(X=t, Y=y, kernel=GPy.kern.RBF(input_dim=1, lengthscale=k2, variance=k1**2))
    mu_gpy, sigma_gpy = model.predict(Xnew=t,include_likelihood=False)

    plt.subplot(3, len(ys), 1 + idx)
    plt.plot(t, f, 'g-', linewidth = 3., label='True function')
    plot_with_uncertainty(x=t, y=mu_f[:N], ystd=np.sqrt(Sigma_f[:N]), label='Unimodal-GP')
    plot_with_uncertainty(x=t, y=mu_gpy, ystd=np.sqrt(sigma_gpy), color='b', fill=False, linestyle='--', label = 'Vanilla-GP')
    plt.plot(t, y, 'k.', markersize=10, label='Data')
    # plt.legend()
    plt.grid(True)

    plt.subplot(3, len(ys),  1 + len(ys) + idx)
    pz = phi(mu_g[:M]/np.sqrt(1 + Sigma_g[:M]))
    plt.plot(t2, pz)
    plt.grid(True)

    plt.subplot(3, len(ys),  1 + 2*len(ys) + idx)
    plot_with_uncertainty(x=t2, y=mu_g[:M], ystd=np.sqrt(Sigma_g[:M]))
    plt.grid(True)

    plt.pause(1e-3)
    plt.show(block = False)