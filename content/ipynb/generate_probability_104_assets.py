import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import os

# Create directory for images if it doesn't exist
IMAGE_DIR = "/home/cjduan/drclab.github.io/static/images/probability-104"
os.makedirs(IMAGE_DIR, exist_ok=True)

def sample_means(data, sample_size):
    means = []
    for _ in range(10_000):
        sample = np.random.choice(data, size=sample_size)
        means.append(np.mean(sample))
    return np.array(means)

def plot_kde_and_qq(sample_means_data, mu_sample_means, sigma_sample_means, title_prefix, filename_prefix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # KDE Plot
    x_range = np.linspace(min(sample_means_data), max(sample_means_data), 100)
    sns.histplot(sample_means_data, stat="density", label="Histogram", ax=ax1)
    sns.kdeplot(data=sample_means_data, color="crimson", label="KDE", linestyle="dashed", fill=True, ax=ax1)
    ax1.plot(x_range, norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means), color="black", label="Gaussian Theoretical")
    ax1.set_title(f"{title_prefix} - Sample Means Distribution")
    ax1.legend()
    
    # QQ Plot
    stats.probplot(sample_means_data, plot=ax2, fit=True)
    ax2.set_title(f"{title_prefix} - QQ Plot")
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, f"{filename_prefix}_kde_qq.png"))
    plt.close()

# 1. Gaussian Population
mu, sigma = 10, 5
gaussian_population = np.random.normal(mu, sigma, 100_000)

plt.figure(figsize=(8, 5))
sns.histplot(gaussian_population, stat="density")
plt.title("Gaussian Population Distribution")
plt.savefig(os.path.join(IMAGE_DIR, "gaussian_population.png"))
plt.close()

n_gauss = 5
gaussian_sample_means = sample_means(gaussian_population, sample_size=n_gauss)
plot_kde_and_qq(gaussian_sample_means, mu, sigma/np.sqrt(n_gauss), "Gaussian", "gaussian_n5")

# 2. Binomial Population
n_bin, p_bin = 5, 0.8
binomial_population = np.random.binomial(n_bin, p_bin, 100_000)

plt.figure(figsize=(8, 5))
sns.histplot(binomial_population, stat="count")
plt.title("Binomial Population Distribution (n=5, p=0.8)")
plt.savefig(os.path.join(IMAGE_DIR, "binomial_population.png"))
plt.close()

# n_sample = 3
n_s3 = 3
bin_s3_means = sample_means(binomial_population, sample_size=n_s3)
plot_kde_and_qq(bin_s3_means, n_bin*p_bin, np.sqrt(n_bin*p_bin*(1-p_bin))/np.sqrt(n_s3), "Binomial (n=3)", "binomial_n3")

# n_sample = 30
n_s30 = 30
bin_s30_means = sample_means(binomial_population, sample_size=n_s30)
plot_kde_and_qq(bin_s30_means, n_bin*p_bin, np.sqrt(n_bin*p_bin*(1-p_bin))/np.sqrt(n_s30), "Binomial (n=30)", "binomial_n30")

# 3. Poisson Population
mu_poi = 1.5
poisson_population = np.random.poisson(mu_poi, 100_000)

plt.figure(figsize=(8, 5))
sns.histplot(poisson_population, stat="density")
plt.title("Poisson Population Distribution (mu=1.5)")
plt.savefig(os.path.join(IMAGE_DIR, "poisson_population.png"))
plt.close()

n_poi = 30
poisson_sample_means = sample_means(poisson_population, sample_size=n_poi)
plot_kde_and_qq(poisson_sample_means, mu_poi, np.sqrt(mu_poi)/np.sqrt(n_poi), "Poisson (n=30)", "poisson_n30")

# 4. Cauchy Population
cauchy_population = np.random.standard_cauchy(1000)

plt.figure(figsize=(8, 5))
# Filtering extreme values for better visualization in population plot
filtered_cauchy = cauchy_population[(cauchy_population > -20) & (cauchy_population < 20)]
sns.histplot(filtered_cauchy, stat="density")
plt.title("Cauchy Population Distribution (Filtered [-20, 20])")
plt.savefig(os.path.join(IMAGE_DIR, "cauchy_population.png"))
plt.close()

# Cauchy sample means n=30
n_c30 = 30
cauchy_s30_means = sample_means(cauchy_population, sample_size=n_c30)
plt.figure(figsize=(6, 6))
stats.probplot(cauchy_s30_means, plot=plt, fit=True)
plt.title("Cauchy (n=30) - QQ Plot")
plt.savefig(os.path.join(IMAGE_DIR, "cauchy_n30_qq.png"))
plt.close()

# Cauchy sample means n=100
n_c100 = 100
cauchy_s100_means = sample_means(cauchy_population, sample_size=n_c100)
plt.figure(figsize=(6, 6))
stats.probplot(cauchy_s100_means, plot=plt, fit=True)
plt.title("Cauchy (n=100) - QQ Plot")
plt.savefig(os.path.join(IMAGE_DIR, "cauchy_n100_qq.png"))
plt.close()

print(f"Assets generated in {IMAGE_DIR}")
