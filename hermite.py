import numpy as np
from scipy.special import poch, hyp2f1, gammaln
from math import factorial
from scipy.integrate import quad
from scipy.special import hermite




# Hn=hermite(n)
#
# Hk=hermite(k)
#

# def f(x):
#     return np.exp(-p*x**2)*Hn(alpha*x)*Hk(beta*x)
#
# # Compute the integral from -inf to inf
# result, error = quad(f, -np.inf, np.inf)
# print(f"result={result}, error={error}")

# Don't create Hermite polynomial objects for large k!
# Hk = hermite(k)  # This causes overflow

def compute_Vnk_even(n, k, lmd, p):
    """Compute Vnk for even n and k using log-space arithmetic"""

    # Compute log(|Ink|)
    log_Ink = 0.5 * np.log(np.pi / p)
    log_Ink += (n + k) * np.log(2)
    log_Ink += ((n + k) / 2) * np.log(np.abs(lmd))

    # Pochhammer symbols in log space
    log_Ink += gammaln(1/2 + n/2) - gammaln(1/2)
    log_Ink += gammaln(1/2 + k/2) - gammaln(1/2)

    # Hypergeometric function
    hyp_val = hyp2f1(-n/2, -k/2, 1/2, -(1-lmd**2)/lmd**2)
    log_Ink += np.log(np.abs(hyp_val))

    # Sign
    sign_Ink = (-1)**int(k/2) * np.sign(hyp_val)

    # Compute log(normalization factor)
    log_norm = -0.5 * ((n + k) * np.log(2) + gammaln(n+1) + gammaln(k+1) + np.log(np.pi))

    # Total
    log_Vnk = log_norm + log_Ink
    sign_Vnk = sign_Ink

    return log_Vnk, sign_Vnk,log_Ink

def compute_Vnk_odd(n, k, lmd, p):
    """Compute Vnk for odd n and k using log-space arithmetic"""

    # Compute log(|Ink|)
    log_Ink = 0.5 * np.log(np.pi / p)
    log_Ink += (n + k - 1) * np.log(2)
    log_Ink += 0.5 * np.log(1 - lmd**2)
    log_Ink += ((n + k) / 2 - 1) * np.log(np.abs(lmd))

    # Pochhammer symbols in log space
    log_Ink += gammaln(3/2 + (n-1)/2) - gammaln(3/2)
    log_Ink += gammaln(3/2 + (k-1)/2) - gammaln(3/2)

    # Hypergeometric function
    hyp_val = hyp2f1(-(n-1)/2, -(k-1)/2, 3/2, -(1-lmd**2)/lmd**2)
    log_Ink += np.log(np.abs(hyp_val))

    # Sign
    sign_Ink = (-1)**int((k-1)/2) * np.sign(hyp_val)

    # Compute log(normalization factor)
    log_norm = -0.5 * ((n + k) * np.log(2) + gammaln(n+1) + gammaln(k+1) + np.log(np.pi))

    # Total
    log_Vnk = log_norm + log_Ink
    sign_Vnk = sign_Ink

    return log_Vnk, sign_Vnk,log_Ink

a = 0.1
p = (a**2 + 1) / 2

alpha = a
beta = 1

lmd = (a**2 - 1) / (a**2 + 1)
# k=20
n = 10


kValsAll=range(0,40000,2)

V_vals=[]
for k in kValsAll:
    log_Vnk, sign_Vnk,_=compute_Vnk_even(n,k,lmd,p)
    V_vals.append(sign_Vnk*np.exp(log_Vnk))

print(V_vals)
log_V_vals=np.array(V_vals)
print(np.linalg.norm(V_vals,ord=2))