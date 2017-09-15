import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.misc import logsumexp

from probit_moments import ProbitMoments
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf, derivLogCdfNormal, logCdfNormal



phi = lambda x: norm.cdf(x)
npdf = lambda x, m, v: 1./np.sqrt(2*np.pi*v)*np.exp(-(x-m)**2/(2*v))



def compute_moments_softinformation(mf, vf, mg, vg, n_std=6, nu2 = 1.):

    def tilted_marginalized_f(g):
        """ Return m0, m1, m2 given by
            
            m0 = int phi( phi(g) * fp )*N(fp | mf, vf)*N(g | mg, vg) df
            m1 = int fp * phi( phi(g) * fp )*N(fp | mf, vf)*N(g | mg, vg) df
            m2 = int fp**2 * phi( phi(g) * fp )*N(fp | mf, vf)*N(g | mg, vg) df
            
        """
        
        k = (2*phi(g)-1)*nu2
        z = k*mf/(np.sqrt(1 + vf*k**2))
        Z, nz = phi(z), npdf(z, 0, 1)
        
        mean = mf + k*vf*nz/(Z*np.sqrt(1 + vf*k**2))
        mean2 = 2*mf*mean - mf**2 + vf - vf**2*z*nz/(Z*(1./k**2 + vf))
        
        m0 = Z*npdf(g, mg, vg)
        m1 = m0*mean #Z*mean*npdf(g, mg, vg)
        m2 = m0*mean2 #Z*mean2*npdf(g, mg, vg)
        
        return m0, m1, m2
    
    n_std = 6.
    g_lower, g_upper = mg - n_std*np.sqrt(vg), mg + n_std*np.sqrt(vg)
    
    # normalization constant
    Z = quad(lambda g: tilted_marginalized_f(g)[0], g_lower, g_upper)[0]
    
    # Moments wrt. fp
    Ef = quad(lambda g: tilted_marginalized_f(g)[1], g_lower, g_upper)[0]/Z
    Ef2 = quad(lambda g: tilted_marginalized_f(g)[2], g_lower, g_upper)[0]/Z
    
    # Moments wrt. g
    Eg = quad(lambda g: g*tilted_marginalized_f(g)[0], g_lower, g_upper)[0]/Z
    Eg2 = quad(lambda g: g**2*tilted_marginalized_f(g)[0], g_lower, g_upper)[0]/Z
    
    return Z, Ef, Ef2, Eg, Eg2
    


def compute_moments_strict(mf, vf, mg, vg, nu2 = 1.):

    #############################################################################
    # Direct implementation (old)
    #############################################################################
    # Z_fp, m1_fp, m2_fp = ProbitMoments.compute_moments(m=0, v=1./nu2, mu=mf, sigma2=vf, return_normalizer=True, normalized=False)
    # Z_g, m1_g, m2_g = ProbitMoments.compute_moments(m=0, v=1, mu=mg, sigma2=vg, return_normalizer=True, normalized=False)
    # Z = (1-Z_fp)*(1-Z_g) + Z_fp*Z_g
     
    # site_fp_m = ((mf-m1_fp)*(1-Z_g) + m1_fp*Z_g)/Z
    # site_fp_m2 = (((mf**2 + vf)-m2_fp)*(1-Z_g) + m2_fp*Z_g)/Z
    # site_g_m = ((1-Z_fp)*(mg-m1_g) + Z_fp*m1_g)/Z
    # site_g_m2 = ((1-Z_fp)*((mg**2 + vg)-m2_g) + m2_g*Z_fp)/Z


    #############################################################################
    # compute moments for fp terms
    #############################################################################
    # Z_fp, m1_fp, m2_fp = ProbitMoments.compute_moments(m=0, v=1./nu2, mu=mf, sigma2=vf, return_normalizer=True, normalized=False)

    v = 1./nu2

    z_f = (mf)/(v*np.sqrt(1 + vf/v**2))

            
    logZ_f = logCdfNormal(z_f)
    logZ_f1m = logCdfNormal(-z_f)
    Z_fp = np.exp(logZ_f)

    phi_div_Phi_f = derivLogCdfNormal(z_f)
    mean_f = mf + vf/(v*np.sqrt(1 + vf/v**2))*phi_div_Phi_f #mu + sigma2*nz/(Z*v*np.sqrt(1 + sigma2/v**2))
    mean2_f = 2*mf*mean_f - mf**2 + vf - vf**2*z_f/((v**2 + vf))*phi_div_Phi_f # sigma2**2*z*nz/(Z*(v**2 + sigma2))

    #############################################################################
    # compute moments for g terms
    #############################################################################
    # Z_g, m1_g, m2_g = ProbitMoments.compute_moments(m=0, v=1, mu=mg, sigma2=vg, return_normalizer=True, normalized=False)

    v = 1.

    z_g = mg/(v*np.sqrt(1 + vg/v**2))

    logZ_g = logCdfNormal(z_g)
    logZ_g1m = logCdfNormal(-z_g)
    Z_g = np.exp(logZ_g)

    phi_div_Phi_g = derivLogCdfNormal(z_g)
    mean_g = mg + vg/(v*np.sqrt(1 + vg/v**2))*phi_div_Phi_g #mu + sigma2*nz/(Z*v*np.sqrt(1 + sigma2/v**2))
    mean2_g = 2*mg*mean_g - mg**2 + vg - vg**2*z_g/((v**2 + vg))*phi_div_Phi_g # sigma2**2*z*nz/(Z*(v**2 + sigma2))

    #############################################################################
    # combine
    #############################################################################
    
    # compute log normalizer: logZ = log[(1-Z_fp)*(1-Z_g) + Z_fp*Z_g]
    log_a1 = logZ_f1m + logZ_g1m
    log_a2 = logZ_f + logZ_g
    logZ = logsumexp((log_a1, log_a2))

    Z = np.exp(logZ)

    # momements
    reciprocal_pa = ((1-Z_fp) + Z_fp*Z_g/(1-Z_g))
    reciprocal_pb = ((1-Z_g) + Z_fp*Z_g/(1-Z_fp))

    site_fp_m = mean_f + (mf - mean_f)/reciprocal_pa
    site_fp_m2 = mean2_f + (mf**2 + vf - mean2_f)/reciprocal_pa

    site_g_m = mean_g + (mg - mean_g)/reciprocal_pb
    site_g_m2 = mean2_g + (mg**2 + vg - mean2_g)/reciprocal_pb


    return Z, site_fp_m, site_fp_m2, site_g_m, site_g_m2

