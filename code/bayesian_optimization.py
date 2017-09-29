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

from importlib import reload
import unimodal 
reload(unimodal)

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
            opt = minimize(acq_n, x, method='L-BFGS-B', bounds = tuple((self.bounds[i,0], self.bounds[i,1]) for i in range(self.dim) ), jac=True, tol=1e-10)
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

    def __init__(self, func_id, func, acquisition_function, bounds=None, max_iter=100, noise = 0.0, g_constraints=False):

        self.g_constraints = g_constraints

        super(UnimodalBayesianOptimization, self).__init__(func_id, func, acquisition_function, bounds, max_iter, noise)



    def get_model(self, X, Y, gp=None):
        if gp is None:
            ker_const = GPy.kern.Bias(input_dim=self.dim, variance=0.5)
            ker_const.variance.constrain_fixed(value=0.5, warning=True, trigger_parent=True)

            ker_sexp = GPy.kern.RBF(input_dim=self.dim, variance=0.1, lengthscale=1.0, ARD=True)
            ker_sexp.lengthscale.set_prior(GPy.priors.HalfT(1,1))
            ker_sexp.variance.set_prior(GPy.priors.HalfT(1,1))
            f_kernel_base = ker_const + ker_sexp
            
            g_variance, g_lengthscale = 0.1, 0.1
            
            g_kernel_base = GPy.kern.RBF(input_dim = self.dim, lengthscale=g_lengthscale, variance=g_variance)
            g_kernel_base.variance.set_prior(GPy.priors.LogGaussian(1., 0.5))
            g_kernel_base.lengthscale.set_prior(GPy.priors.LogGaussian(-1, 0.1))

            lik = GPy.likelihoods.Gaussian(variance=self.noise)
            if self.noise < 0.00001:
                lik.variance.constrain_fixed(value=1e-8,warning=True,trigger_parent=True)
            else:
                prior = GPy.priors.InverseGamma(3,0.25)
                lik.variance.set_prior(prior)


            
            if self.dim == 1:
                M = 10
                Xd = np.linspace(0, 1, M)[:, None]
            else:
                M = 7
                x1 = np.linspace(0, 1, M)
                x2 = np.linspace(0, 1, M)
                X1, X2 = np.meshgrid(x1, x2)
                Xd = np.column_stack((X1.ravel(), X2.ravel()))
                assert(self.dim == 2)

            if self.g_constraints:

                if self.dim == 1:
                    Xq = [np.column_stack((Xd[[0, -1]], np.zeros((2, 1))))]
                    Yq = [np.array([-1, 1])[:, None]]
                else:


                    Xq = []
                    Yq = []

                    Q = 5

                    xq = np.linspace(0, 1, Q)

                    # dim 1
                    E1 = np.column_stack((np.zeros((Q)), xq, np.zeros((Q))))
                    E2 = np.column_stack((np.ones((Q)), xq, np.zeros((Q))))
                    E = np.row_stack((E1, E2))

                    W1 = -1*np.ones((len(E1), 1))
                    W2 = np.ones((len(E1), 1))
                    W = np.row_stack((W1, W2))

                    Xq.append(E)
                    Yq.append(W)


                    # dim 2
                    E1 = np.column_stack((xq, np.zeros((Q)), np.zeros((Q))))
                    E2 = np.column_stack((xq, np.ones((Q)), np.zeros((Q))))
                    E = np.row_stack((E1, E2))

                    W1 = -1*np.ones((len(E1), 1))
                    W2 = np.ones((len(E1), 1))
                    W = np.row_stack((W1, W2))

                    Xq.append(E)
                    Yq.append(W)




# 
                    # import ipdb; ipdb.set_trace()


                    # Xq = None
                    # Yq = None
                    
            else:
                Xq = None
                Yq = None

            self.model = unimodal.UnimodalGP(X=X, Y=Y, Xd=Xd, f_kernel_base=f_kernel_base, g_kernel_base=g_kernel_base, likelihood=lik, Xq=Xq, Yq=Yq)

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
    max_itt = 0


    noise = 0.05
    # Define test functions
    l = test_function_base.get_gaussian_functions(20,2,1)
    l_new0 = test_function_base.normalize_functions(l)
    l_new = test_function_base.noisify_functions(l_new0, noise)        

    
    num_probs = 10
    f_idx = 0

    # print("BO with vanilla GP")
    # bo = BayesianOptimization(func_id=0, func = l_new[0][f_idx], acquisition_function=EI, max_iter=max_itt, noise = noise)
    # X,Y = bo.optimize()
    # path=root+'vanilla/'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # bo.save_metrics(path)

    print("BO with unimodal GP")
    uni_bo = UnimodalBayesianOptimization(func_id=0, func = l_new[0][f_idx], acquisition_function=EI, max_iter=max_itt, noise = noise, g_constraints=True)
    Xu, Yu = uni_bo.optimize()


    fig = plt.figure(figsize = (20, 10))


    for itt in range(10):

        fig.clf()

        xs = np.linspace(-0.5, 1.5, 101)

        Xu, Yu = uni_bo.X, uni_bo.Y


        mu, var = uni_bo.model.predict(xs)
        mu_g, var_g = uni_bo.model.sample_z_probabilities(xs)

        # acq
        self = uni_bo
        preds, _ = self.model.predict(self.X)
        acq_n = lambda x: self.acq(np.array([x]), fmin = min(preds), model = self.model, n=self.X.shape[0], d=self.dim)
        vals = -np.vstack([acq_n(xi)[0] for xi in xs])
        x_new = self.maximize_acquisition()

        # plot
        plt.subplot(2, 2, 1)
        plot_with_uncertainty(xs, mu, var, color='r')
        plt.plot(xs, np.stack([l_new0[0][f_idx].do_evaluate(xi) for xi in xs]))
        plt.plot(Xu, Yu, 'k.')
        for bound in uni_bo.bounds[0]:
            plt.axvline(bound, color='k', linestyle='--', alpha=0.5)

        plt.grid(True)
        plt.title('Iteration {}'.format(itt))



        plt.subplot(2, 2, 2)
        plot_with_uncertainty(xs, mu_g, var_g, color='r')
        plt.grid(True)
        plt.ylim((-0.1, 1.1))

        # plot acq
        plt.subplot(2, 2, 3)
        plt.plot(xs, vals)
        plt.title('Acquisition function')
        plt.grid(True)

        for bound in uni_bo.bounds[0]:
            plt.axvline(bound, color='k', linestyle='--', alpha=0.5)

        plt.axvline(x_new, color='g', alpha = 0.5)
        fig.tight_layout()

        plt.pause(1e-2)


        plt.show(block = False)


        self._add_point(x_new, force=False)



    # path = root +'vanilla/'
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # uni_bo.save_metrics(path)
    
    # xs = np.linspace(-1, 1, 1001)[:, None]
    # uni_mu, uni_var = uni_bo.model.predict(xs)
    # reg_mu, reg_var = bo.model.predict(xs)

    # plt.subplot(1, 2, 1)
    # plot_with_uncertainty(xs, uni_mu, np.sqrt(uni_var), color='r', label='Unimodal')
    # plt.plot(Xu, Yu, 'r.', markersize=10)

    # plot_with_uncertainty(xs, reg_mu, np.sqrt(reg_var), color='b', label='Regular')
    # ytrue = [l_new0[0][f_idx].do_evaluate(xi) for xi in xs]
    # plt.plot(xs, ytrue, 'g--', linewidth = 2)

    # plt.plot(X, Y, 'b.')
    # plt.grid(True)
    # plt.legend()

    # plt.ylim([-0.5, 1.5])


    # plt.subplot(1, 2, 2)
    # plt.plot(Yu, 'r', label = 'Unimodal')
    # plt.plot(Y, 'b', label = 'Regular')
    # plt.ylim([-0.5, 1.5])
    # plt.legend()
    # plt.grid(True)
    # plt.xlabel('Iterations')
    # plt.show()

