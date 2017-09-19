import numpy as np

import GPy
from GPy.likelihoods import Gaussian
import paramz

from copy import deepcopy

from importlib import reload

import ep_unimodality as ep
reload(ep)

class UnimodalGP(GPy.core.Model):


    def __init__(self, X, Y, Xd, f_kernel_base, g_kernel_base, likelihood, name='UnimodalGP'):

        super(UnimodalGP, self).__init__(name=name)

        self.N, self.D = X.shape
        self.M = Xd.shape[0]

        # store data
        self.X = X
        self.Y = Y
        self.Xd = Xd

        ###################################################################################
        # Likelihood
        ###################################################################################
        # Fixed hyperparameters
        # self.sigma2 = GPy.core.parameterization.Param('Noise variance', sigma2)
        # self.sigma2.constrain_positive()
        # self.link_parameter(self.sigma2)
        assert(isinstance(likelihood, Gaussian)), "Only Gaussian likelihood is supported!"
        self.likelihood = likelihood
        self.link_parameter(self.likelihood)

        ###################################################################################
        # Contruct kernel for f
        ###################################################################################
        self.f_kernel_base = f_kernel_base

        # construct lists of kernel for f and fprime for each dimension
        f_kernel_list = [self.f_kernel_base] + [GPy.kern.DiffKern(self.f_kernel_base, d) for d in range(self.D)]    
        t_list = [X] + [Xd.copy() for d in range(self.D)]
        y_dummy_list = [None] + [None for d in range(self.D)]

        # TODO: Remove y_dummy_list
        self.Xf, _, self.Xf_output_index = GPy.util.multioutput.build_XY(t_list, y_dummy_list)
        self.Kf_kernel = GPy.kern.MultioutputKern(kernels=f_kernel_list, cross_covariances={}, name='Kf')

        self.link_parameter(self.Kf_kernel)

        ###################################################################################
        # Contruct kernel for each g
        ###################################################################################
        self.g_kernel_base = g_kernel_base

        self.Kg_kernel_list = []
        for d in range(self.D):
            g_kernel = self.g_kernel_base.copy()
            g_kernel_der = GPy.kern.DiffKern(g_kernel, d)
            Kg_kernel = GPy.kern.MultioutputKern(kernels=[g_kernel, g_kernel_der], cross_covariances={}, name='Kg%d'%d)
            self.Kg_kernel_list.append(Kg_kernel)
            self.link_parameter(self.Kg_kernel_list[d])

        self.Xg, _, self.Xg_output_index = GPy.util.multioutput.build_XY([Xd, Xd], [None, None])

    def parameters_changed(self):

        # Run EP
        self.f_posterior, self.g_posterior_list, Kf, self._log_lik, self.grad_dict = ep.ep_unimodality(self.Xf, self.Xg, self.X, self.Y, Kf_kernel=self.Kf_kernel.copy(), Kg_kernel_list=self.Kg_kernel_list, sigma2=self.likelihood.variance, t2=self.Xd, verbose=0, nu2=1., tol=1e-10, max_itt=100)

        # update gradients for noise variance
        self.likelihood.variance.gradient = np.sum(np.diag(self.grad_dict['dL_dK_f'])[:self.N])

        # update gradients for f
        self.Kf_kernel.update_gradients_full(self.grad_dict['dL_dK_f'], self.Xf)

        # update gradients for each g
        for d in range(self.D):
            self.Kg_kernel_list[d].update_gradients_full(self.grad_dict['dL_dK_g%d' % d], self.Xg)


    def log_likelihood(self):
        return self._log_lik

    def predict(self, Xnew, full_cov=False, Y_metadata=None, include_likelihood=True):

        if Y_metadata is not None:
            print('Provided meta data is not used!')

        # augment Xnew with kernel index
        Xp = np.column_stack(  (Xnew, np.zeros((len(Xnew), 1))) )

        # predict
        pred_mean, pred_var = self.f_posterior._raw_predict(self.Kf_kernel, Xp, self.Xf, full_cov=full_cov)

        if include_likelihood:
            pred_var = pred_var + self.likelihood.variance

        return pred_mean, pred_var


    def predict_g(self, Xnew, g_index=0, full_cov=False):

        # augment Xnew with kernel index
        Xp = np.column_stack(  (Xnew, np.zeros((len(Xnew), 1))) )

        # predict
        pred_mean, pred_var = self.g_posterior_list[g_index]._raw_predict(self.Kg_kernel_list[g_index], Xp, self.Xg, full_cov=full_cov)

        return pred_mean, pred_var
    
    def predictive_gradients(self, Xnew):
        pred_mean, pred_cov =  self.predict_g(Xnew)
        return np.reshape(pred_mean, (pred_mean.shape[0], self.D,1)), pred_cov
        
    def sample_z_probabilities(self, Xnew, g_index=0, num_samples=1000):

        pred_mean, pred_cov = self.predict_g(Xnew, g_index=g_index, full_cov=True)
        D = pred_cov.shape[0]

        L = np.linalg.cholesky(pred_cov + 1e-6*np.identity(D)) 

        zs = pred_mean + np.dot(L, np.random.normal(0, 1, size=(D, num_samples)))
        pzs = ep.phi(zs)

        return np.mean(pzs, axis = 1), np.var(pzs, axis = 1)    


    def log_predictive_density(self, Xtest, ytest, Y_metadata=None):
        
        if Y_metadata is not None:
            print('Provided meta data is not used!')
        mu_test, var_test = self.predict(Xtest)
        return ep.log_npdf(ytest, mu_test[:, None], var_test[:, None])

    def set_XY(self, X=None, Y=None, Xd=None):
        self.N, self.D = X.shape
        if(Xd is not None):
            self.M = Xd.shape[0]
            self.Xd = Xd
        self.update_model(False)
        if Y is not None:
            self.Y = paramz.ObsAr(Y)
            self.Y_normalized = self.Y
        if X is not None:
            self.X = paramz.ObsAr(X)
        self.update_model(True)








