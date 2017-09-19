import GPy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from functools import partial
from scipy.special import erfc
from scipy.optimize import minimize
from get_factorial import get_factorial
import copy
import time
import test_function_base
from ep_unimodality import phi
import unimodal 

from util import plot_with_uncertainty

def get_quantiles(fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations. 
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin-m)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))    
    return (phi, Phi, u)

def EI(x, fmin=None, model=None, n=None, d=None):
    m, v = model.predict(x)
    v = np.clip(v, 1e-10, np.inf)
    s = np.sqrt(v)
    phi, Phi, u = get_quantiles(fmin, m, s)
    f_acqu = s * (u * Phi + phi)
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*np.sqrt(v))
    df_acqu = dsdx * phi - Phi * dmdx
    return -f_acqu, -df_acqu

def LCB(x, fmin=None, model = None, n=None, d=None):
    m, v = model.predict(x)
    v = np.clip(v, 1e-10, np.inf)
    s = np.sqrt(v)
    eta = 0.1
    exploration_weight = np.sqrt(2.*np.log((n**(d/2.+2.))*(np.pi**2.)/(3.*eta)))
    f_acqu = -m + exploration_weight * s
    
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*s)
    df_acqu = -dmdx + exploration_weight * dsdx
    return -f_acqu, -df_acqu

def PI(x, fmin=None, model = None, n=None, d=None):
    m, v = model.predict(x)
    v = np.clip(v, 1e-10, np.inf)
    s = np.sqrt(v)
    phi, Phi, u = get_quantiles(fmin, m, s)    
    f_acqu = Phi

    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*np.sqrt(v))    
    df_acqu = -(phi/s)* (dmdx + dsdx * u)
    return -f_acqu, -df_acqu

class BayesianOptimization(object):
    def __init__(self, func_id, func, acquisition_function, bounds=None, max_iter=100, noise = 0.0):
        self.func_id = func_id
        self.func = func
        self.acq = acquisition_function
        self.dim = self.func.dim
        self.max_iter = max_iter
        self.noise=noise
        self.reset()
        if bounds is None:
            bounds = np.array([[0,1] for i in range(self.dim)])
        self.bounds = bounds
        
        self.f_min = []
        self.f_min_ref = []
        self.x_min = []
    
    def collect_metrics(self):
        preds, _ = self.model.predict(self.X)
        ind = np.argmin(preds)
        self.f_min = self.f_min + [preds.reshape((-1))[ind]]
        self.x_min = self.x_min + [ind]
        
    def save_metrics(self, folder):
        x_min = self.X[self.x_min,:]
        f_min_ref = np.array([self.func.evaluate_clean(x_min[i,:]) for i in range(x_min.shape[0])])
        tmp = {'f_min': self.f_min, 'x_min': self.x_min, 'x':self.X, 'f_min_ref':f_min_ref, 'x_min_real':self.func.min_loc, 'f_id': self.func_id}
        pickle.dump(tmp, open(folder + str(self.func_id) + ".p", "wb" ) )
    
    def reset(self):
        self.X, self.Y = self.get_XY();
        self.get_model(self.X, self.Y)
        
    def get_XY(self):
        x = get_factorial(self.dim)*3./10. + 0.5
        x = np.append(x, 0.5*np.ones((1, self.dim)), axis=0)
        y = np.array([self.func.do_evaluate(x[i,:]) for i in range(x.shape[0])]).reshape((-1,1))
        return x, y
    
    def _add_point(self, x_new, force=False):
        self.X = np.append(self.X, x_new, axis=0)
        self.Y = np.append(self.Y, np.array([[self.func.do_evaluate(x_new)]]), axis=0)
        self.get_model(self.X, self.Y)
    
    def _get_size(self):
        return self.X.shape[0]
    
    def get_model(self, X, Y, noise=0., gp=None):
        if gp is None:
            ker_const = GPy.kern.Bias(input_dim=self.dim, variance=0.5)
            ker_const.variance.constrain_fixed(value=0.5, warning=True, trigger_parent=True)

            ker_sexp = GPy.kern.RBF(input_dim=self.dim, variance=0.1, lengthscale=1.0, ARD=True)
            prior = GPy.priors.HalfT(1,1)
            ker_sexp.lengthscale.set_prior(prior)
            prior = GPy.priors.HalfT(1,1)
            ker_sexp.variance.set_prior(prior)
            ker = ker_const + ker_sexp

            lik = GPy.likelihoods.Gaussian(variance=self.noise)
            if self.noise < 0.00001:
                lik.variance.constrain_fixed(value=1e-8,warning=True,trigger_parent=True)
            else:
                prior = GPy.priors.InverseGamma(3,0.25)
                lik.variance.set_prior(prior)
            self.model = GPy.core.GP(X = X, Y = Y, kernel=ker, likelihood=lik)
        else:
            self.model.set_XY(X, Y)
        self.model.optimize()
    
    def maximize_acquisition(self, num_points = 10):
        x_best = None
        preds, _ = self.model.predict(self.X)
        acq_n = lambda x: self.acq(np.array([x]), fmin = min(preds), model = self.model, n=self.X.shape[0], d=self.dim)
        best = np.inf
        for i in range(num_points):
            x = np.random.rand(1,self.dim)
            opt = minimize(acq_n, x, method='L-BFGS-B', bounds = tuple((self.bounds[i,0], self.bounds[i,1]) for i in range(self.dim) ), jac=True, tol=1e-50)
            temp,_ = acq_n(opt.x)
            if temp < best:
                x_best = opt.x
                best = temp
        return np.array([x_best])
    
    def optimize(self):
        np.random.seed(self.func_id)
        for i in range(self.max_iter):
            print("Iteration {}".format(i))
            n = self._get_size()
            while self._get_size() == n:
                start = time.time()
                x_new = self.maximize_acquisition()
                end = time.time()
                print("Maximizing acq. took: {}".format(str(end-start)))
                start = time.time()
                self._add_point(x_new, force=False)
                end = time.time()
                print("Adding point to the model took: {}".format(str(end-start)))
            print(self.model)
            self.collect_metrics()
        return self.X, self.Y
    
class UnimodalBayesianOptimization(BayesianOptimization):

    def get_model(self, X, Y, gp=None):
        if gp is None:
            ker_const = GPy.kern.Bias(input_dim=self.dim, variance=0.5)
            ker_const.variance.constrain_fixed(value=0.5, warning=True, trigger_parent=True)

            ker_sexp = GPy.kern.RBF(input_dim=self.dim, variance=0.1, lengthscale=1.0, ARD=True)
            prior = GPy.priors.HalfT(1,1)
            ker_sexp.lengthscale.set_prior(prior)
            prior = GPy.priors.HalfT(1,1)
            ker_sexp.variance.set_prior(prior)
            f_kernel_base = ker_const + ker_sexp
            
            g_variance, g_lengthscale = 1., 1.
            
            g_kernel_base = GPy.kern.RBF(input_dim = self.dim, lengthscale=g_lengthscale, variance=g_variance)
            g_kernel_base.variance.set_prior(GPy.priors.HalfT(1,1))
            g_kernel_base.lengthscale.set_prior(GPy.priors.HalfT(1,1))

            lik = GPy.likelihoods.Gaussian(variance=self.noise)
            if self.noise < 0.00001:
                lik.variance.constrain_fixed(value=1e-8,warning=True,trigger_parent=True)
            else:
                prior = GPy.priors.InverseGamma(3,0.25)
                lik.variance.set_prior(prior)

            M = 20
            Xd = np.linspace(0, 1, M)[:, None]
            
            self.model = unimodal.UnimodalGP(X=X, Y=Y, Xd=Xd, f_kernel_base=f_kernel_base, g_kernel_base=g_kernel_base, likelihood=lik)
        else:
            self.model.set_XY(X, Y)
        start = time.time()
        self.model.optimize()
        end = time.time()
        print("Optimizing GP took: {}".format(str(end-start)))


if __name__ == "__main__":
    import os
    root = './results/'
    if not os.path.exists(root):
        os.mkdir(root)

    np.random.seed(0)
    max_itt = 10

    # Define test functions
    l = test_function_base.get_gaussian_functions(20,2,1)
    l_new0 = test_function_base.normalize_functions(l)
    l_new = test_function_base.noisify_functions(l_new0, 0.05)        

    print("BO with vanilla GP")
    bo = BayesianOptimization(func_id=0, func = l_new[0][0], acquisition_function=EI, max_iter=max_itt, noise = 0.05)
    X,Y = bo.optimize()
    path=root+'vanilla/'
    if not os.path.exists(path):
        os.mkdir(path)
    bo.save_metrics(path)


    print("BO with unimodal GP")
    uni_bo = UnimodalBayesianOptimization(func_id=0, func = l_new[0][0], acquisition_function=EI, max_iter=max_itt, noise = 0.05)
    Xu,Yu = uni_bo.optimize()
    path = root +'vanilla/'
    if not os.path.exists(path):
        os.mkdir(path)
    uni_bo.save_metrics(path)
    
    xs = np.linspace(0, 1, 1001)[:, None]
    uni_mu, uni_var = uni_bo.model.predict(xs)
    reg_mu, reg_var = bo.model.predict(xs)

    plt.subplot(1, 2, 1)
    plot_with_uncertainty(xs, uni_mu, np.sqrt(uni_var), color='r', label='Unimodal')
    plt.plot(Xu, Yu, 'r.', markersize=10)

    plot_with_uncertainty(xs, reg_mu, np.sqrt(reg_var), color='b', label='Regular')
    ytrue = [l_new0[0][0].do_evaluate(xi) for xi in xs]
    plt.plot(xs, ytrue, 'g--', linewidth = 2)

    plt.plot(X, Y, 'b.')
    plt.grid(True)
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(Yu, 'r', label = 'Unimodal')
    plt.plot(Y, 'b', label = 'Regular')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.show()

