import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

# Load the data
demo_data = np.loadtxt("line.csv", delimiter=",")
demo_positions = demo_data.T  # Assuming positions are stored column-wise

# Fit Bayesian Gaussian Mixture Model
max_components = 10  # Upper bound for components
bgmm = BayesianGaussianMixture(n_components=max_components, covariance_type='full', weight_concentration_prior_type='dirichlet_process', random_state=42)
bgmm.fit(demo_positions)

# Find the effective number of components (i.e., those with significant weights)
weights = bgmm.weights_
optimal_n_components = np.sum(weights > 1e-2)  # Count components with non-negligible weight

print(f"Optimal number of components (determined by BGMM): {optimal_n_components}")

# Scatter plot of data points
plt.scatter(demo_positions[:, 0], demo_positions[:, 1], alpha=0.5, label="Data")

# Scatter plot of GMM means
plt.scatter(bgmm.means_[:, 0], bgmm.means_[:, 1], color="red", marker="x", label="BGMM Means")

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Bayesian GMM Clustering")
plt.legend()
plt.show()