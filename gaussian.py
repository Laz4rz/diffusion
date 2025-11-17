import numpy as np

# hyperparameters
alpha1 = 0.8
alpha2 = 0.7
sigma1 = np.sqrt(1 - alpha1)  # e.g. like diffusion
sigma2 = np.sqrt(1 - alpha2)

def posterior_x1_given_x0_x2(x0, x2):
    """
    Compute mean and variance of p(x1 | x2, x0)
    for the 1D chain:
        x1 | x0 ~ N(sqrt(alpha1)*x0, sigma1^2)
        x2 | x1 ~ N(sqrt(alpha2)*x1, sigma2^2)
    """
    m1 = np.sqrt(alpha1) * x0
    
    A = 1.0 / sigma1**2 + alpha2 / sigma2**2
    B = m1 / sigma1**2 + (np.sqrt(alpha2) * x2) / sigma2**2
    
    var_post = 1.0 / A
    mean_post = B / A
    return mean_post, var_post

# simulate many samples and compare
rng = np.random.default_rng(0)
N = 200_000

# sample x0
x0 = rng.normal(0.0, 1.0, size=N)

# forward: x1 | x0
x1 = np.sqrt(alpha1) * x0 + sigma1 * rng.normal(size=N)

# forward: x2 | x1
x2 = np.sqrt(alpha2) * x1 + sigma2 * rng.normal(size=N)

# choose a particular (x0*, x2*) to condition on (approximate)
x0_star = 0.5
x2_star = -1.0

# get indices where (x0, x2) are near (x0_star, x2_star)
mask = (np.abs(x0 - x0_star) < 0.02) & (np.abs(x2 - x2_star) < 0.02)
x1_cond_samples = x1[mask]

print("Number of approx. conditional samples:", x1_cond_samples.size)

# empirical mean/var
emp_mean = x1_cond_samples.mean()
emp_var = x1_cond_samples.var()

# analytic mean/var
post_mean, post_var = posterior_x1_given_x0_x2(x0_star, x2_star)

print("Empirical mean:", emp_mean)
print("Analytic mean:", post_mean)
print("Empirical var :", emp_var)
print("Analytic var  :", post_var)
