import numpy as np
from scipy.special import poch, hyp2f1, gammaln
from math import factorial
from scipy.integrate import quad
from scipy.special import hermite
from scipy.special import eval_hermite
from datetime import datetime
import matplotlib.pyplot as plt


#numerical test of Hermite function scaling


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

def hermite_basis(n, x):
    """
    Compute the normalized Hermite function u_n(x).

    u_n(x) = 1/sqrt(2^n * n! * sqrt(pi)) * exp(-x^2/2) * H_n(x)

    Parameters:
    -----------
    n : int
        Order of the Hermite function
    x : float or array
        Point(s) at which to evaluate the function

    Returns:
    --------
    float or array
        Value of u_n(x)
    """
    # Use log-space for normalization to avoid overflow
    log_norm = -0.5 * (n * np.log(2) + gammaln(n+1) + 0.5 * np.log(np.pi))
    norm = np.exp(log_norm)
    return norm * np.exp(-0.5 * x**2) * eval_hermite(n, x)

def compute_VE(n_max, a):
    """
    Compute the even part submatrix V^E.

    V^E[i,j] corresponds to V_{2i, 2j} from the full matrix.

    Parameters:
    -----------
    n_max : int
        Maximum index for even Hermite functions (will compute up to u_{2*n_max})
    a : float
        Scaling parameter

    Returns:
    --------
    V_E : ndarray of shape (n_max, n_max)
        Even part submatrix
    """
    p = (a**2 + 1) / 2
    lmd = (a**2 - 1) / (a**2 + 1)

    V_E = np.zeros((n_max, n_max))

    for i in range(n_max):
        for j in range(n_max):
            n = 2 * j  # Column index in full matrix
            k = 2 * i  # Row index in full matrix
            log_Vnk, sign_Vnk,_=compute_Vnk_even(n, k, lmd, p)
            V_E[i, j] =sign_Vnk*np.exp(log_Vnk)

    return V_E

def compute_VO(n_max, a):
    """
    Compute the odd part submatrix V^O.

    V^O[i,j] corresponds to V_{2i+1, 2j+1} from the full matrix.

    Parameters:
    -----------
    n_max : int
        Maximum index for odd Hermite functions (will compute up to u_{2*n_max+1})
    a : float
        Scaling parameter

    Returns:
    --------
    V_O : ndarray of shape (n_max, n_max)
        Odd part submatrix
    """
    p = (a**2 + 1) / 2
    lmd = (a**2 - 1) / (a**2 + 1)

    V_O = np.zeros((n_max, n_max))

    for i in range(n_max):
        for j in range(n_max):
            n = 2 * j + 1  # Column index in full matrix
            k = 2 * i + 1  # Row index in full matrix
            log_Vnk, sign_Vnk,_=compute_Vnk_odd(n, k, lmd, p)
            V_O[i, j] = sign_Vnk*np.exp(log_Vnk)


    return V_O


def construct_function(b_coeffs, x, eps=1e-15):
    """
    Construct a function f(x) = sum_n b_n * u_n(x)

    Parameters:
    -----------
    b_coeffs : array-like
        Coefficients [b_0, b_1, b_2, ...]
    x : float or array
        Point(s) at which to evaluate the function
    eps : float
        Threshold for including coefficients (only use |b_n| > eps)

    Returns:
    --------
    float or array
        Value of f(x)
    """
    result = 0.0
    for n, b_n in enumerate(b_coeffs):
        if np.abs(b_n) > eps:
            result += b_n * hermite_basis(n, x)
    return result


def compute_scaled_coefficients(b_coeffs, a, n_max_E, n_max_O):
    """
    Compute coefficients for f(ax) given coefficients of f(x).

    If f(x) = sum_n b_n * u_n(x), then
    f(ax) = sum_k c_k * u_k(x)

    where c = V * b (using the appropriate even/odd submatrices)

    Parameters:
    -----------
    b_coeffs : array-like
        Coefficients [b_0, b_1, b_2, ...] for f(x)
    a : float
        Scaling parameter
    n_max_E : int
        Dimension for V^E matrix
    n_max_O : int
        Dimension for V^O matrix

    Returns:
    --------
    c_coeffs : ndarray
        Coefficients for f(ax)
    """
    b_coeffs = np.array(b_coeffs)

    # Separate even and odd coefficients
    b_even = b_coeffs[0::2]  # b_0, b_2, b_4, ...
    b_odd = b_coeffs[1::2]   # b_1, b_3, b_5, ...

    # Pad if necessary
    if len(b_even) < n_max_E:
        b_even = np.pad(b_even, (0, n_max_E - len(b_even)))
    if len(b_odd) < n_max_O:
        b_odd = np.pad(b_odd, (0, n_max_O - len(b_odd)))

    # Compute V^E and V^O
    V_E = compute_VE(n_max_E, a)
    V_O = compute_VO(n_max_O, a)

    # Transform coefficients
    c_even = V_E @ b_even  # c_0, c_2, c_4, ...
    c_odd = V_O @ b_odd    # c_1, c_3, c_5, ...

    # Interleave back into full coefficient array
    n_total = 2 * max(n_max_E, n_max_O)
    c_coeffs = np.zeros(n_total)
    c_coeffs[0::2] = c_even
    c_coeffs[1::2] = c_odd

    return c_coeffs

a = 0.99
p = (a**2 + 1) / 2

alpha = a
beta = 1

lmd = (a**2 - 1) / (a**2 + 1)


n_max_E=100
n_max_O=100

b_coeffs=[1,1,]

t_interpolation_start=datetime.now()
c_coeffs=compute_scaled_coefficients(b_coeffs,a,n_max_E,n_max_O)

t_interpolation_end=datetime.now()
print(f"interpolation time: ",t_interpolation_end-t_interpolation_start)

# Test points
x_test = np.linspace(-3, 3, 200)

# Compute f(ax) directly
f_ax_direct = construct_function(b_coeffs, a * x_test)


# Compute f(ax) using transformed coefficients
f_ax_transformed = construct_function(c_coeffs, x_test)


# Compare
error = np.abs(f_ax_direct - f_ax_transformed)
max_error = np.max(error)
print(f"Maximum error: {max_error}")
print(f"Mean error: {np.mean(error)}")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Direct vs Transformed
axes[0, 0].plot(x_test, f_ax_direct, 'b-', label='f(ax) direct', linewidth=2)
axes[0, 0].plot(x_test, f_ax_transformed, 'r--', label='f(ax) transformed', linewidth=2)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('f(ax)')
axes[0, 0].set_title('Comparison of Direct vs Transformed')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Error
axes[0, 1].plot(x_test, error, 'g-', linewidth=2)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Absolute Error')
axes[0, 1].set_title(f'Error (max = {max_error:.2e})')
axes[0, 1].grid(True)
axes[0, 1].set_yscale('log')

# Plot 3: Coefficient magnitudes
n_coeffs_to_plot = min(50, len(c_coeffs))
axes[1, 0].semilogy(range(n_coeffs_to_plot), np.abs(c_coeffs[:n_coeffs_to_plot]), 'bo-', markersize=4)
axes[1, 0].set_xlabel('Coefficient index')
axes[1, 0].set_ylabel('|c_n|')
axes[1, 0].set_title('Magnitude of Transformed Coefficients')
axes[1, 0].grid(True)

# Plot 4: Original coefficients vs significant transformed coefficients
axes[1, 1].semilogy(range(len(b_coeffs)), np.abs(b_coeffs), 'ro-', markersize=8, label='Original b_n')
threshold = 1e-10 * np.max(np.abs(c_coeffs))
significant_indices = np.where(np.abs(c_coeffs) > threshold)[0]
axes[1, 1].semilogy(significant_indices, np.abs(c_coeffs[significant_indices]), 'bs', markersize=4, label='Transformed c_n (significant)')
axes[1, 1].set_xlabel('Coefficient index')
axes[1, 1].set_ylabel('|coefficient|')
axes[1, 1].set_title('Original vs Transformed Coefficients')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('hermite_scaling_results.png', dpi=150)
# plt.show()
plt.close()
print(f"\nNumber of significant transformed coefficients: {len(significant_indices)}")
print(f"Significant indices: {significant_indices[:20]}...")  # Show first 20