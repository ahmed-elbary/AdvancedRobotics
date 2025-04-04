import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import matplotlib.patches as patches

# Load the data
demo_data = np.loadtxt("Line.csv", delimiter=",")
demo_positions = demo_data.T  # Assuming positions are stored column-wise

n_components_range = range(1, 11)  # Test from 1 to 10 components
likelihoods = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(demo_positions)
    likelihoods.append(gmm.score(demo_positions))  # Log-likelihood of the dataset

# Plot Log-Likelihood
plt.figure(figsize=(6, 4))
plt.plot(n_components_range, likelihoods, marker="o", linestyle="-")
plt.xlabel("Number of Components")
plt.ylabel("Log-Likelihood")
plt.title("GMM Log-Likelihood vs. Number of Components")
plt.show()

# Select optimal GMM model
optimal_n_components = n_components_range[np.argmax(likelihoods)]
best_gmm = GaussianMixture(n_components=optimal_n_components, covariance_type='full', random_state=42)
best_gmm.fit(demo_positions)

# Plot Data and GMM Means
plt.figure(figsize=(6, 6))
plt.scatter(demo_positions[:, 0], demo_positions[:, 1], label="Data", alpha=0.5)
plt.scatter(best_gmm.means_[:, 0], best_gmm.means_[:, 1], color="red", marker="x", label="GMM Means")

# Plot Covariance Ellipses
def plot_cov_ellipse(mean, cov, ax, color="red"):
    """Plot a covariance ellipse."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.arctan2(*eigenvectors[:, 0][::-1])
    width, height = 2 * np.sqrt(eigenvalues)
    
    ellipse = patches.Ellipse(mean, width, height, angle=np.degrees(angle), edgecolor=color, facecolor='none', linewidth=2)
    ax.add_patch(ellipse)

ax = plt.gca()
for i in range(optimal_n_components):
    plot_cov_ellipse(best_gmm.means_[i], best_gmm.covariances_[i], ax)

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("GMM Clustering with Covariance Ellipses")
plt.legend()
plt.show()

print(f"Optimal number of components: {optimal_n_components}")
