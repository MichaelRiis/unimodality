import numpy as np
import time
from scipy.integrate import quad, dblquad
from scipy.stats import norm
from scipy.misc import logsumexp

import GPy
from GPy.inference.latent_function_inference.expectation_propagation import posteriorParams
from GPy.util.linalg import  dtrtrs, dpotrs, tdot, symmetrify, jitchol

from probit_moments import ProbitMoments
from moment_functions import compute_moments_softinformation, compute_moments_strict
from util import mult_diag

npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))
log_npdf = lambda x, m, v: -0.5*np.log(2*np.pi*v) -(x-m)**2/(2*v)
phi = lambda x: norm.cdf(x)
logphi = lambda x: norm.logcdf(x)



def update_posterior(K, eta, theta):
    D = K.shape[0]
    sqrt_theta = np.sqrt(theta)
    G = sqrt_theta[:, None]*K
    B = np.identity(D) + G*sqrt_theta
    L = jitchol(B)
    V = np.linalg.solve(L, G)
    Sigma_full = K - np.dot(V.T, V)
    mu = np.dot(Sigma_full, eta)
    #Sigma = np.diag(Sigma_full)

    return posteriorParams(mu=mu, Sigma=Sigma_full, L=L)

    # return mu, Sigma, Sigma_full, L

def ep_unimodality(X1, X2, t, y, Kf_kernel, Kg_kernel_list, sigma2, t2=None, m=None, max_itt=50, nu=10., nu2 = 1., alpha=0.9, tol=1e-6, verbose=0, moment_function=None, seed=0):

    np.random.seed(seed)
    t0 = time.time()

    if t2 is None:
        t2 = t.copy()


    N, D = t.shape
    M = len(t2)
    Df = N + D*M
    Dg = M + M

    if m is None:
        m = np.ones((D, M))

    # moment function
    if moment_function is None:
        moment_function = compute_moments_strict


    ###################################################################################
    # Initialize sites
    ###################################################################################

    # sites for likelihood 
    eta_y, theta_y = np.zeros(Df), np.zeros(Df)
    eta_y[:N], theta_y[:N] = y[:, 0]/sigma2, 1./sigma2

    # sites connecting f' and g
    initial_site_variance = 1e-16
    eta_fp, theta_fp = np.zeros(Df), initial_site_variance*np.ones(Df)
    eta_g, theta_g = np.array([np.zeros(Dg) for d in range(D)]), np.array([initial_site_variance*np.ones(Dg) for d in range(D)])

    # sites for g' and m
    eta_gp, theta_gp =np.array([np.zeros(Dg) for d in range(D)]) , np.array([initial_site_variance*np.ones(Dg) for d in range(D)])

    ###################################################################################
    # Contruct kernels
    ###################################################################################
    Kf = Kf_kernel.K(X1)
    Kg_list = [kg.K(X2) for kg in Kg_kernel_list]

    ###################################################################################
    # Prepare global approximations
    ###################################################################################
    f_posterior = update_posterior(Kf, eta_fp + eta_y, theta_fp + theta_y)
    g_posterior_list = [update_posterior(Kg_list[d], eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d]) for d in range(D)]




    ###################################################################################
    # Iterate
    ###################################################################################
    for itt in range(max_itt):

        old_params = np.hstack((f_posterior.mu, f_posterior.Sigma_diag)) # , mu_g, Sigma_g

        if verbose > 0:
            print('Iteration %d' % (itt + 1))

        # approximate constraints to enforce monotonicity to g
        d_list = np.random.choice(range(D), size=D, replace=False)
        for d in d_list:

            g_posterior = g_posterior_list[d]

            j_list = np.random.choice(range(M), size=M, replace=False) if M > 0 else []
            for j in j_list:

                # compute offset for gradient indices
                i = M + j

                # compute cavity
                eta_cav, theta_cav = g_posterior.mu[i]/g_posterior.Sigma_diag[i] - eta_gp[d, j], 1./g_posterior.Sigma_diag[i] - theta_gp[d, j]
                m_cav, v_cav = eta_cav/theta_cav, 1./theta_cav

                if v_cav <= 0:
                    print('EP: Negative cavity observed at site %d in dim %d in iteration %d, skipping update' % (j, d, itt + 1))
                    continue

                # compute moments
                Z, site_m, site_m2 = ProbitMoments.compute_moments(m=0, v=1./(m.ravel()[j]*nu), mu=m_cav, sigma2=v_cav, return_normalizer=True, normalized=True)

                # variance
                site_v = site_m2 - site_m**2

                new_eta = site_m/site_v - eta_cav
                new_theta = 1./site_v - theta_cav

                if new_theta < 0:
                    new_theta = 1e-6
                    new_variance = 1./(new_theta + theta_cav)
                    new_eta = site_m/new_variance - eta_cav

                # update site
                eta_gp[d, i], theta_gp[d, i] = (1-alpha)*eta_gp[d, i] + alpha*new_eta, (1-alpha)*theta_gp[d, i] + alpha*new_theta

            # update joint
            g_posterior_list[d] = update_posterior(Kg_list[d], eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d])

      # approximate constraints to enforce a single sign change for f'
        d_list = np.random.choice(range(D), size=D, replace=False)
        for d in d_list:

            g_posterior = g_posterior_list[d]
            j_list = np.random.choice(range(M), size=M, replace=False) if M > 0 else []
            for j in j_list:

                i = N + d*M +  j

                # compute cavity
                eta_cav_fp, theta_cav_fp = f_posterior.mu[i]/f_posterior.Sigma_diag[i] - eta_fp[j], 1./f_posterior.Sigma_diag[i] - theta_fp[j]
                eta_cav_g, theta_cav_g = g_posterior.mu[j]/g_posterior.Sigma_diag[j] - eta_g[d, j], 1./g_posterior.Sigma_diag[j] - theta_g[d, j]

                # transform to means and variances
                m_cav_fp, v_cav_fp = eta_cav_fp/theta_cav_fp, 1./theta_cav_fp
                m_cav_g, v_cav_g = eta_cav_g/theta_cav_g, 1./theta_cav_g

                if v_cav_fp <= 0 or v_cav_g <= 0:
                    print('Negative cavity variance for site %d! Skipping...' % j)
                    continue

                # compute moments
                Z, site_fp_m, site_fp_m2, site_g_m, site_g_m2 = moment_function(m_cav_fp, v_cav_fp, m_cav_g, v_cav_g, nu2=nu2)

                if Z == 0 or np.isnan(Z):
                    print('Z = 0 occured, skipping...')
                    continue

                # variances
                site_fp_v = site_fp_m2 - site_fp_m**2
                site_g_v = site_g_m2 - site_g_m**2

                # new sites
                new_eta_fp, new_theta_fp = site_fp_m/site_fp_v - eta_cav_fp, 1./site_fp_v - theta_cav_fp
                new_eta_g, new_theta_g = site_g_m/site_g_v - eta_cav_g, 1./site_g_v - theta_cav_g

                if new_theta_fp <= 0:
                    new_theta_fp = 1e-6
                    new_variance_fp = 1./(new_theta_fp + theta_cav_fp)
                    new_eta_fp = site_fp_m/new_variance_fp - eta_cav_fp

                if new_theta_g <= 0:
                    new_theta_g = 1e-6
                    new_variance_g = 1./(new_theta_g + theta_cav_g)
                    new_eta_g = site_g_m/new_variance_g - eta_cav_g


                # update site
                eta_fp[i] = (1-alpha)*eta_fp[i] + alpha*new_eta_fp
                theta_fp[i] = (1-alpha)*theta_fp[i] + alpha*new_theta_fp

                eta_g[d, j] = (1-alpha)*eta_g[d, j] + alpha*new_eta_g
                theta_g[d, j] = (1-alpha)*theta_g[d, j] + alpha*new_theta_g

            # update posterior
            g_posterior_list[d] = update_posterior(Kg_list[d], eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d])
            f_posterior = update_posterior(Kf, eta_fp + eta_y, theta_fp + theta_y)

      # check for convergence
        new_params = np.hstack((f_posterior.mu, f_posterior.Sigma_diag)) # , mu_g, Sigma_g
        if len(old_params) > 0 and np.mean((new_params-old_params)**2)/np.mean(old_params**2) < tol:
            run_time = time.time() - t0

            if verbose > 0:
                print('Converged in %d iterations in %4.3fs' % (itt + 1, run_time))
            break

    #############################################################################3
    # Marginal likelihood
    #############################################################################3

    # multivariate terms likelihood
    f_term = compute_marginal_likelihood_mvn(f_posterior, eta_fp + eta_y, theta_fp + theta_y, skip_problematic=N)
    g_terms = [compute_marginal_likelihood_mvn(g_posterior, eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d], skip_problematic=0)  for d, g_posterior in zip(range(D), g_posterior_list)]

    log_k1, log_k2 = 0, 0
    log_c1, log_c2, log_c3, log_c4 = 0, 0, 0, 0

    for d in range(D):

        mu_g, Sigma_g = g_posterior_list[d].mu, g_posterior_list[d].Sigma_diag
        eta_cav, theta_cav = mu_g[M:]/Sigma_g[M:] - eta_gp[d, M:], 1./Sigma_g[M:] - theta_gp[d, M:]
        mu_cav, tau_cav = eta_cav/theta_cav, 1./theta_cav

        # log k_i
        log_k1 += np.sum(ProbitMoments.compute_normalization(m=0, v=1./(nu*m[d, :]), mu=mu_cav, sigma2= tau_cav, log=True))
        log_k2_prob = 0 #*0.5*np.sum(-np.log(theta_gp[:, M:]))
        log_k2 += log_k2_prob + 0.5*np.sum(np.log(1 + theta_gp[d, M:]/theta_cav)) + 0.5*np.sum((mu_cav - eta_gp[d, M:]/theta_gp[d, M:])**2/(tau_cav + 1./theta_gp[d, M:]))

        # log c_i
        fp_slice = slice(N + d*M, N + (d+1)*M)
        eta_cav_fp, theta_cav_fp = f_posterior.mu[fp_slice]/f_posterior.Sigma_diag[fp_slice] - eta_fp[fp_slice], 1./f_posterior.Sigma_diag[fp_slice] - theta_fp[fp_slice]
        eta_cav_g, theta_cav_g = mu_g[:M]/Sigma_g[:M] - eta_g[d, :M], 1./Sigma_g[:M] - theta_g[d, :M]

        m_cav_fp, v_cav_fp = eta_cav_fp/theta_cav_fp, 1./theta_cav_fp
        m_cav_g, v_cav_g = eta_cav_g/theta_cav_g, 1./theta_cav_g

        # compute expectation of mixture site wrt. cavity
        log_A1 = ProbitMoments.compute_normalization(m=0, v=-1./nu2, mu=m_cav_fp, sigma2=v_cav_fp, log=True)
        log_A2 = ProbitMoments.compute_normalization(m=0, v=-1, mu=m_cav_g, sigma2=v_cav_g, log=True)
        log_A3 = ProbitMoments.compute_normalization(m=0, v=1./nu2, mu=m_cav_fp, sigma2=v_cav_fp, log=True)
        log_A4 = ProbitMoments.compute_normalization(m=0, v=1., mu=m_cav_g, sigma2=v_cav_g, log=True)

        log_c1 += np.sum(logsumexp(np.row_stack((log_A1 + log_A2, log_A3 + log_A4)), axis = 0, keepdims=True))
        log_c2 += 0

        # problematic terms
        log_c3_prob = 0#0*0.5*np.sum(-np.log(theta_fp[N:]))
        log_c4_prob = 0#0*0.5*np.sum(-np.log(theta_g[:, :M]))

        log_c3 += log_c3_prob + 0.5*np.sum(np.log(1 + theta_fp[fp_slice]/theta_cav_fp)) + 0.5*np.sum((m_cav_fp - eta_fp[fp_slice]/theta_fp[fp_slice])**2/(v_cav_fp + 1./theta_fp[fp_slice]))
        log_c4 += log_c4_prob + 0.5*np.sum(np.log(1 + theta_g[:, :M]/theta_cav_g)) + 0.5*np.sum((m_cav_g - eta_g[d, :M]/theta_g[d, :M])**2/(v_cav_g + 1./theta_g[d, :M]))

    logZ = log_k1 + log_k2 + log_c1 + log_c2 + log_c3 + log_c4 +  f_term + np.sum(g_terms)

    #############################################################################3
    # handle gradients for f and each g
    #############################################################################3
    grad_dict = {'dL_dK_f': compute_dl_dK(f_posterior, Kf, eta_fp + eta_y, theta_fp + theta_y)}
    
    for d in range(D):
        grad_dict['dL_dK_g%d' % d] = compute_dl_dK(g_posterior_list[d], Kg_list[d], eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d])

    # Done
    return f_posterior, g_posterior_list, Kf, logZ, grad_dict#, mu_g, Sigma_g, Sigma_full_g, logZ

def compute_dl_dK(posterior, K, eta, theta, prior_mean = 0):
    tau, v = theta, eta

    tau_tilde_root = np.sqrt(tau)
    Sroot_tilde_K = tau_tilde_root[:,None] * K
    aux_alpha , _ = dpotrs(posterior.L, np.dot(Sroot_tilde_K, v), lower=1)
    alpha = (v - tau_tilde_root * aux_alpha)[:,None] #(K + Sigma^(\tilde))^(-1) /mu^(/tilde)
    LWi, _ = dtrtrs(posterior.L, np.diag(tau_tilde_root), lower=1)
    Wi = np.dot(LWi.T, LWi)
    symmetrify(Wi) #(K + Sigma^(\tilde))^(-1)

    dL_dK = 0.5 * (tdot(alpha) - Wi)
    
    return dL_dK


def compute_marginal_likelihood_mvn(posterior, eta, theta, skip_problematic=None):

    mu, Sigma = posterior.mu, posterior.Sigma
    
    b = np.linalg.solve(posterior.L, eta/np.sqrt(theta))

    # skip problematic term that will cancel out later?
    if skip_problematic is None:
        problematic_term = - np.sum(np.log(np.sqrt(theta)))
    elif skip_problematic == 0:
        problematic_term = 0
    elif skip_problematic > 0:
        problematic_term = - np.sum(np.log(np.sqrt(theta[:skip_problematic])))

    logdet = np.sum(np.log(np.diag(posterior.L))) + problematic_term
    quadterm = 0.5*np.sum(b**2)

    return -0.5*len(mu)*np.log(2*np.pi)  - logdet - quadterm