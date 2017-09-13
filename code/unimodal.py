import numpy as np

import GPy
import paramz

from copy import deepcopy
import ep_unimodality_2d as ep


class UnimodalGP(GPy.core.Model):


    def __init__(self, X, Y, Xd, f_kernel_base, g_kernel_base, sigma2, name='UnimodalGP'):

        super(UnimodalGP, self).__init__(name=name)

        self.N, self.D = X.shape
        self.M = Xd.shape[0]

        # store data
        self.X = X
        self.Y = Y
        self.Xd = Xd

        # Fixed hyperparameters
        self.sigma2 = sigma2

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
        self.mu_f, Sigma_f, self.Sigma_full_f, self.g_posterior_list, Kf, self._log_lik, self.grad_dict = ep.ep_unimodality(self.Xf, self.Xg, self.X, self.Y, Kf_kernel=self.Kf_kernel.copy(), Kg_kernel_list=self.Kg_kernel_list, sigma2=self.sigma2, t2=self.Xd, verbose=0, nu2=1., tol=1e-6, max_itt=100)

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

        if full_cov:
            raise NotImplementedError('Fullcov not implemented')

        # augment Xnew with kernel index
        Xp = np.column_stack(  (Xnew, np.zeros((len(Xnew), 1))) )

        # concatenate and augment X + Xd with kernel indeces (equal to self.Xf when predicting f)
        # Xf = np.column_stack(  (self.X, np.zeros((self.N, 1))) )
        # Xf = np.row_stack([Xf] + [np.column_stack(  (self.Xd, (d+1)*np.ones((self.M, 1))) ) for d in range(self.D)])

        # construct kernels
        Kff = self.Kf_kernel.K(self.Xf)
        Kpp = self.Kf_kernel.K(Xp, Xp)
        Kpf = self.Kf_kernel.K(Xp, self.Xf)

        # Compute predictive distributions
        H =  np.linalg.solve(Kff, Kpf.T)
        pred_mean = np.dot(H.T, self.mu_f)
        pred_cov = Kpp -  np.dot(Kpf, H) + np.dot(H.T, np.dot(self.Sigma_full_f, H))

        # if not full_cov:
        pred_var_ = np.diag(pred_cov)

        if include_likelihood:
            pred_var = pred_var_ + self.sigma2

        return pred_mean, pred_var


    def predict_g(self, Xnew, g_index=0, full_cov=False):
        mu_g, Sigma_g, Sigma_full_g, Lg = self.g_posterior_list[g_index]

        # augment Xnew with kernel index
        Xp = np.column_stack(  (Xnew, np.zeros((len(Xnew), 1))) )

        # concatenate and augment Xd + Xd with kernel indeces (equal to self.Xf when predicting f)
        Xg = np.column_stack(  (self.Xd, np.zeros((self.M, 1))) )
        Xg = np.row_stack([Xg] + [np.column_stack(  (self.Xd, np.ones((self.M, 1))) )])

        # construct kernels
        Kg_kernel = self.Kg_kernel_list[g_index]
        Kgg = Kg_kernel.K(Xg)
        Kpp = Kg_kernel.K(Xp, Xp)
        Kpg = Kg_kernel.K(Xp, Xg)

        # Compute predictive distributions
        H =  np.linalg.solve(Kgg, Kpg.T)
        pred_mean = np.dot(H.T, mu_g)
        pred_cov = Kpp -  np.dot(Kpg, H) + np.dot(H.T, np.dot(Sigma_full_g, H))

        if not full_cov:
            pred_cov = np.diag(pred_cov)


        return pred_mean, pred_cov


    def sample_z_probabilities(self, Xnew, g_index=0, num_samples=1000):

        pred_mean, pred_cov = self.predict_g(Xnew, g_index=g_index, full_cov=True)
        D = pred_cov.shape[0]

        L = np.linalg.cholesky(pred_cov + 1e-6*np.identity(D)) 

        zs = pred_mean[:, None] + np.dot(L, np.random.normal(0, 1, size=(D, num_samples)))
        pzs = ep.phi(zs)

        return np.mean(pzs, axis = 1), np.var(pzs, axis = 1)    


    def log_predictive_density(self, Xtest, ytest, Y_metadata=None):
        
        if Y_metadata is not None:
            print('Provided meta data is not used!')

        mu_test, var_test = self.predict(Xtest)
        return ep.log_npdf(ytest, mu_test[:, None], var_test[:, None] + self.sigma2)










