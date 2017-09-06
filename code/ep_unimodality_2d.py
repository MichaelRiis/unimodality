import numpy as np
import time
from scipy.integrate import quad, dblquad
from scipy.stats import norm

from scipy.misc import logsumexp


from probit_moments import ProbitMoments
from moment_functions import compute_moments_softinformation, compute_moments_strict

from derivative_kernels import generate_joint_derivative_kernel, cov_fun0, cov_fun1, cov_fun2

npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))
log_npdf = lambda x, m, v: -0.5*np.log(2*np.pi*v) -(x-m)**2/(2*v)
phi = lambda x: norm.cdf(x)
logphi = lambda x: norm.logcdf(x)


def update_posterior(K, eta, theta):
    D = K.shape[0]
    sqrt_theta = np.sqrt(theta)
    G = sqrt_theta[:, None]*K
    B = np.identity(D) + G*sqrt_theta
    L = np.linalg.cholesky(B)
    V = np.linalg.solve(L, G)
    Sigma_full = K - np.dot(V.T, V)
    mu, Sigma = np.dot(Sigma_full, eta), np.diag(Sigma_full)

    return mu, Sigma, Sigma_full, L

def ep_unimodality(t, y, k1, k2, sigma2, t2=None, m=None, max_itt=50, nu=10., nu2 = 1., alpha=0.9, tol=1e-4, verbose=0, c1=None, c2=None, moment_function=None, seed=0, k3=0):

    np.random.seed(seed)
    t0 = time.time()

    if t2 is None:
        t2 = t.copy()


    N, D = t.shape
    M = len(t2)
    Df = N + D*M
    Dg = M + M

    print(Df, Dg)

    if m is None:
        m = np.ones((D, M))

    print('Number of derivative points per dim: {}'.format(M))

    # Hyperparameter for g
    if c1 is None:
        c1 = k1
    if c2 is None:
        c2 = k2

    # moment function
    if moment_function is None:
        moment_function = compute_moments_strict



    # sites for likelihood 
    eta_y, theta_y = np.zeros(Df), np.zeros(Df)
    eta_y[:N], theta_y[:N] = y[:, 0]/sigma2, 1./sigma2

    # sites connecting f' and g
    initial_site_variance = 1e-16
    eta_fp, theta_fp = np.zeros(Df), initial_site_variance*np.ones(Df)
    eta_g, theta_g = np.array([np.zeros(Dg) for d in range(D)]), np.array([initial_site_variance*np.ones(Dg) for d in range(D)])

    # sites for g' and m
    eta_gp, theta_gp =np.array([np.zeros(Dg) for d in range(D)]) , np.array([initial_site_variance*np.ones(Dg) for d in range(D)])

    # # prepare kernels
    t_grad_list = [t2.copy() for i in range(D)]
    Kf = generate_joint_derivative_kernel(t, t_grad_list, k1, k2, k3)


    # t_grad_list1 = [t2.copy(), None, None]
    # t_grad_list2 = [None, t2.copy(), None]
    # t_grad_list3 = [None, None, t2.copy()]
    # t_grad_list = [t_grad_list1, t_grad_list2, t_grad_list3]


    t_grad_list = []
    for d1 in range(D):
        t_grad_list.append([t2.copy() if d2 is d1 else None for d2 in range(D)])
# 


    Kg_list = [generate_joint_derivative_kernel(t2.copy(), t_grad_list_i, c1, c2) for t_grad_list_i in t_grad_list]




    print('N: {}'.format(N))
    print('M: {}'.format(M))
    print('D: {}'.format(D))
    print('N + D*M: {}'.format(N + D*M))

    print(Kf.shape)
    for Kg in Kg_list:
        print(Kg.shape)



    # prepare global approximation
    mu_f, Sigma_f, Sigma_full_f, Lf = update_posterior(Kf, eta_fp + eta_y, theta_fp + theta_y)
    g_posterior_list = [update_posterior(Kg_list[d], eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d]) for d in range(D)]


    for itt in range(max_itt):

        old_params = np.hstack((mu_f, Sigma_f)) # , mu_g, Sigma_g

        if verbose > 0:
            print('Iteration %d' % (itt + 1))

        # approximate constraints to enforce monotonicity to g
        d_list = np.random.choice(range(D), size=D, replace=False)
        for d in d_list:

            mu_g, Sigma_g, Sigma_full_g, Lg = g_posterior_list[d]

            j_list = np.random.choice(range(M), size=M, replace=False) if M > 0 else []
            for j in j_list:

                # compute offset for gradient indices
                i = M + j

                # compute cavity
                eta_cav, theta_cav = mu_g[i]/Sigma_g[i] - eta_gp[d, j], 1./Sigma_g[i] - theta_gp[d, j]
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
                    # new_variance = 1./(new_theta + theta_cav)
                    # new_eta = site_m/new_variance - eta_cav

                # update site
                eta_gp[d, i], theta_gp[d, i] = (1-alpha)*eta_gp[d, i] + alpha*new_eta, (1-alpha)*theta_gp[d, i] + alpha*new_theta

            # update joint
            g_posterior_list[d] = update_posterior(Kg_list[d], eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d])



      # approximate constraints to enforce a single sign change for f'
        d_list = np.random.choice(range(D), size=D, replace=False)
        for d in d_list:

            mu_g, Sigma_g, Sigma_full_g, Lg = g_posterior_list[d]
            j_list = np.random.choice(range(M), size=M, replace=False) if M > 0 else []
            for j in j_list:

                i = N + d*M +  j

                # compute cavity
                eta_cav_fp, theta_cav_fp = mu_f[i]/Sigma_f[i] - eta_fp[j], 1./Sigma_f[i] - theta_fp[j]
                eta_cav_g, theta_cav_g = mu_g[j]/Sigma_g[j] - eta_g[d, j], 1./Sigma_g[j] - theta_g[d, j]

                # transform to means and variances
                m_cav_fp, v_cav_fp = eta_cav_fp/theta_cav_fp, 1./theta_cav_fp
                m_cav_g, v_cav_g = eta_cav_g/theta_cav_g, 1./theta_cav_g

                if v_cav_fp <= 0 or v_cav_g <= 0:
                    print('Negative cavity variance for site %d! Skipping...' % j)
                    continue

                # compute moments
                # Z_fp, m1_fp, m2_fp = ProbitMoments.compute_moments(m=0, v=1, mu=m_cav_fp, sigma2=v_cav_fp, return_normalizer=True, normalized=False)
                # Z_g, m1_g, m2_g = ProbitMoments.compute_moments(m=0, v=1, mu=m_cav_g, sigma2=v_cav_g, return_normalizer=True, normalized=False)
                # Z = (1-Z_fp)*(1-Z_g) + Z_fp*Z_g

                # site_fp_m = ((m_cav_fp-m1_fp)*(1-Z_g) + m1_fp*Z_g)/Z
                # site_fp_m2 = (((m_cav_fp**2 + v_cav_fp)-m2_fp)*(1-Z_g) + m2_fp*Z_g)/Z
                # site_g_m = ((1-Z_fp)*(m_cav_g-m1_g) + Z_fp*m1_g)/Z
                # site_g_m2 = ((1-Z_fp)*((m_cav_g**2 + v_cav_g)-m2_g) + m2_g*Z_fp)/Z

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
                    # new_variance_fp = 1./(new_theta_fp + theta_cav_fp)
                    # new_eta_fp = site_fp_m/new_variance_fp - eta_cav_fp

                if new_theta_g <= 0:
                    new_theta_g = 1e-6
                    # new_variance_g = 1./(new_theta_g + theta_cav_g)
                    # new_eta_g = site_g_m/new_variance_g - eta_cav_g


                # update site
                eta_fp[i] = (1-alpha)*eta_fp[i] + alpha*new_eta_fp
                theta_fp[i] = (1-alpha)*theta_fp[i] + alpha*new_theta_fp

                eta_g[d, j] = (1-alpha)*eta_g[d, j] + alpha*new_eta_g
                theta_g[d, j] = (1-alpha)*theta_g[d, j] + alpha*new_theta_g

            # update posterior
            g_posterior_list[d] = update_posterior(Kg_list[d], eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d])
            mu_f, Sigma_f, Sigma_full_f, Lf = update_posterior(Kf, eta_fp + eta_y, theta_fp + theta_y)

      # check for convergence
        new_params = np.hstack((mu_f, Sigma_f)) # , mu_g, Sigma_g
        if len(old_params) > 0 and np.mean((new_params-old_params)**2)/np.mean(old_params**2) < tol:
            run_time = time.time() - t0


            print('Converged in %d iterations in %4.3fs' % (itt + 1, run_time))
            break



    # marginal likelihood
    f_term = compute_marginal_likelihood_mvn(Lf, mu_f, Sigma_full_f, eta_fp + eta_y, theta_fp + theta_y, skip_problematic=N)
    g_terms = [compute_marginal_likelihood_mvn(Lg, mu_g, Sigma_full_g, eta_g[d] + eta_gp[d], theta_g[d] + theta_gp[d], skip_problematic=0)  for d, (mu_g, _, Sigma_full_g, Lg)in zip(range(D), g_posterior_list)]

# 
    # log k_i
    # TODO: DOES NOT WORK FOR the CASE D > 1
    mu_g, _, Sigma_full_g, Lg = g_posterior_list[0]
    eta_cav, theta_cav = mu_g/Sigma_g - eta_gp, 1./Sigma_g - theta_gp
    mu_cav, tau_cav = eta_cav/theta_cav, 1./theta_cav


    log_k1 = np.sum(ProbitMoments.compute_normalization(m=0, v=1./(nu*m), mu=mu_cav[:, M:], sigma2= tau_cav[:, M:], log=True))
    log_k2_prob = 0*0.5*np.sum(-np.log(theta_gp[:, M:]))
    log_k2 = log_k2_prob + 0.5*np.sum(np.log(tau_cav[:, M:]*theta_gp[:, M:] + 1)) + 0.5*np.sum((mu_cav[:, M:] - eta_gp[:, M:]/theta_gp[:, M:])**2/(tau_cav[:, M:] + 1./theta_gp[:, M:]))


    # log c_i
    eta_cav_fp, theta_cav_fp = mu_f/Sigma_f - eta_fp, 1./Sigma_f - theta_fp
    eta_cav_g, theta_cav_g = mu_g/Sigma_g - eta_g, 1./Sigma_g - theta_g
    m_cav_fp, v_cav_fp = eta_cav_fp/theta_cav_fp, 1./theta_cav_fp
    m_cav_g, v_cav_g = eta_cav_g/theta_cav_g, 1./theta_cav_g


    # compute expectation of mixture site wrt. cavity
    log_A1 = ProbitMoments.compute_normalization(m=0, v=-1./nu2, mu=m_cav_fp[N:], sigma2=v_cav_fp[N:], log=True)
    log_A2 = ProbitMoments.compute_normalization(m=0, v=-1, mu=m_cav_g[:, :M], sigma2=v_cav_g[:, :M], log=True)
    log_A3 = ProbitMoments.compute_normalization(m=0, v=1./nu2, mu=m_cav_fp[N:], sigma2=v_cav_fp[N:], log=True)
    log_A4 = ProbitMoments.compute_normalization(m=0, v=1., mu=m_cav_g[:, :M], sigma2=v_cav_g[:, :M], log=True)

    log_c1, log_c2 = np.sum(logsumexp(np.row_stack((log_A1 + log_A2, log_A3 + log_A4)), axis = 0, keepdims=True)), 0
 
    # problematic terms
    log_c3_prob = 0*0.5*np.sum(-np.log(theta_fp[N:]))
    log_c4_prob = 0*0.5*np.sum(-np.log(theta_g[:, :M]))

    log_c3 = log_c3_prob + 0.5*np.sum(np.log(v_cav_fp[N:]*theta_fp[N:] +1)) + 0.5*np.sum((m_cav_fp[N:] - eta_fp[N:]/theta_fp[N:])**2/(v_cav_fp[N:] + 1./theta_fp[N:]))
    log_c4 = log_c4_prob + 0.5*np.sum(np.log(v_cav_g[:, :M]*theta_g[:, :M] + 1)) + 0.5*np.sum((m_cav_g[:, :M] - eta_g[:, :M]/theta_g[:, :M])**2/(v_cav_g[:, :M] + 1./theta_g[:, :M]))

    logZ = log_k1 + log_k2 + log_c1 + log_c2 + log_c3 + log_c4 +  f_term + np.sum(g_terms)
    print('Log Z: %6.5f' % logZ)

# 
    # import ipdb; ipdb.set_trace()
# 


    return mu_f, Sigma_f, Sigma_full_f, g_posterior_list, Kf, logZ #, mu_g, Sigma_g, Sigma_full_g, logZ

def compute_marginal_likelihood_mvn(L, mu, Sigma, eta, theta, skip_problematic=None):
    
    b = np.linalg.solve(L, eta/np.sqrt(theta))

    # skip problematic term that will cancel out later?
    if skip_problematic is None:
        problematic_term = - np.sum(np.log(np.sqrt(theta)))
    elif skip_problematic == 0:
        problematic_term = 0
    elif skip_problematic > 0:
        problematic_term = - np.sum(np.log(np.sqrt(theta[:skip_problematic])))

    logdet = np.sum(np.log(np.diag(L))) + problematic_term
    quadterm = 0.5*np.sum(b**2)

    return -0.5*len(mu)*np.log(2*np.pi)  - logdet - quadterm


def _predict(mu, Sigma_full, t, t_grad_list, t_pred, k1, k2, k3, f=True):
    """ returns predictive mean and full covariance """

    # TODO: Need to be validated!

    # kernel functions
    # cov_fun = lambda x, y: k1**2*np.exp(-0.5*(x-y)**2/k2**2)
    # cov_fun1 = lambda x, y: -cov_fun(x,y)*(x-y)/k2**2
    # cov_fun2 = lambda x, y: cov_fun(x,y)*(1 - (x-y)**2/k2**2 )/k2**2

    if k3 is None:
        k3 = 0

    M = None
    for t_grad in t_grad_list:
        if t_grad is not None:
            M = len(t_grad)

    N, D = t.shape
    # M = 100#len(t2)
    if f:
        Df = N + D*M
    else: 
        Df = N + M
    P = t_pred.shape[0]


    # K = generate_joint_derivative_kernel(t, t2, k1, k2)

    # t_grad_list = [t2.copy() for i in range(t.shape[1])]
    K = generate_joint_derivative_kernel(t, t_grad_list, k1, k2, k3=k3)

    # TODO: Make more flexible
    Ms = [len(t_grad) if t_grad is not None else 0 for t_grad in t_grad_list]



    Kpp = cov_fun0(t_pred, t_pred.T, k1, k2) + k3

    Kpf = np.zeros((P, Df))
    Kpf[:, :N] = cov_fun0(t_pred, t.T, k1, k2) + k3

    offset = N
    for g in range(D):
        if Ms[g] == 0:
            continue
        Kpf[:P, offset:offset + Ms[g]] = cov_fun1(t_grad_list[g], t_pred, g, k1, k2).T
        offset += Ms[g]

    H =  np.linalg.solve(K, Kpf.T)
    pred_mean = np.dot(H.T, mu)

    pred_cov = Kpp -  np.dot(Kpf, H) + np.dot(H.T, np.dot(Sigma_full, H))
    return pred_mean, pred_cov

def predict(mu, Sigma_full, t, t2, t_pred, k1, k2, k3=None, sigma2 = None, f=True):

    pred_mean, pred_cov = _predict(mu, Sigma_full, t, t2, t_pred, k1, k2, k3, f)
    pred_var_ = np.diag(pred_cov)

    if sigma2 is None:
        sigma2 = 0

    pred_var = pred_var_ + sigma2

    return pred_mean, pred_var

def lppd(ytest, mu, Sigma_full, t, t2, t_pred, k1, k2, k3=None, sigma2 = None, per_sample=False):

    pred_mean, pred_var = predict(mu, Sigma_full, t, t2, t_pred, k1, k2, k3, sigma2)

    lppd = log_npdf(ytest.ravel(), pred_mean, pred_var)

    if not per_sample:
        lppd = np.mean(lppd)

    return lppd





def sample_z_probabilities(mu, Sigma_full, t, t2, t_pred, k1, k2, num_samples = 1000):


    pred_mean, pred_cov = _predict(mu, Sigma_full, t, t2, t_pred, k1, k2)
    D = pred_cov.shape[0]

    L = np.linalg.cholesky(pred_cov + 1e-8*np.identity(D)) 

    zs = pred_mean[:, None] + np.dot(L, np.random.normal(0, 1, size=(D, num_samples)))
    pzs = phi(zs)

    return np.mean(pzs, axis = 1), np.var(pzs, axis = 1)

