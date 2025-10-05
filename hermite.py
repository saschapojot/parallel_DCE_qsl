import numpy as np
from scipy.special import poch, hyp2f1, gammaln
from math import factorial

a = 0.1
p = (a**2 + 1) / 2

alpha = a
beta = 1

lmd = (a**2 - 1) / (a**2 + 1)

n = 5
k = 11111

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

    return log_Vnk, sign_Vnk

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

    return log_Vnk, sign_Vnk

# Determine parity and compute
n_even = (n % 2 == 0)
k_even = (k % 2 == 0)

if n_even and k_even:
    print("Computing for even n and k...")
    log_Vnk, sign = compute_Vnk_even(n, k, lmd, p)
elif (not n_even) and (not k_even):
    print("Computing for odd n and k...")
    log_Vnk, sign = compute_Vnk_odd(n, k, lmd, p)
else:
    print("Error: n and k must have the same parity (both even or both odd)")
    print("The integral is zero for mixed parity.")
    exit()

print(f"log(|Vnk|) = {log_Vnk}")
print(f"sign(Vnk) = {sign}")

# If the log value is not too negative, compute the actual value
if log_Vnk > -700:  # Avoid underflow
    Vnk = sign * np.exp(log_Vnk)
    print(f"Vnk = {Vnk}")
else:
    print(f"Vnk is too small to represent as float (< 10^-304)")
    print(f"|Vnk| ≈ 10^{log_Vnk / np.log(10)}")