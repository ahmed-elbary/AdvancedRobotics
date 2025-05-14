import numpy as np
import matplotlib.pyplot as plt
from pydmps.dmp_discrete import DMPs_discrete

# ----- STEP 1: Load and prepare the trajectory -----
demo_folder = "D:/University of Lincoln/Semester B/Advanced Robotics/AdvancedRobotics/Assessment2/datasets/"
demo_file = "SShape.csv"
y_demo = np.loadtxt(demo_folder + demo_file, delimiter=",").T  # Final shape: (150, 2)
print("Fixed trajectory shape:", y_demo.shape)

# ----- STEP 2: RMSE function -----
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.sum((y_true - y_pred) ** 2, axis=1)))

# ----- STEP 3: Evaluate DMPs with different n_bfs -----
n_bfs_list = list(range(10, 301, 10))
rmse_list = []
trajectories = []

# Use only the first (T-1) points for training and RMSE
y_demo_trimmed = y_demo[:-1]
timesteps = len(y_demo_trimmed)
dt = 1.0 / timesteps

for n_bfs in n_bfs_list:
    dmp = DMPs_discrete(n_dmps=2, n_bfs=n_bfs, dt=dt)
    dmp.y0 = y_demo[0]
    dmp.goal = y_demo[-1]
    dmp.timesteps = timesteps

    dmp.imitate_path(y_des=y_demo_trimmed.T)  # Transposed: (2, 149)
    y_pred, _, _ = dmp.rollout()

    rmse = compute_rmse(y_demo_trimmed, y_pred)
    rmse_list.append(rmse)
    trajectories.append(y_pred)

    print(f"n_bfs = {n_bfs:3d} → RMSE = {rmse:.4f}")

# ----- STEP 4: Plot best DMP trajectory vs demonstration -----
best_idx = np.argmin(rmse_list)
best_n_bfs = n_bfs_list[best_idx]
best_pred = trajectories[best_idx]

plt.figure(figsize=(6, 5))
plt.plot(y_demo_trimmed[:, 0], y_demo_trimmed[:, 1], 'k--', label='Demonstration')
plt.plot(best_pred[:, 0], best_pred[:, 1], 'b-', label=f'DMP (n_bfs={best_n_bfs})')
plt.scatter(*y_demo[0], c='g', label='Start')
plt.scatter(*y_demo[-1], c='r', label='Goal')
plt.title(f"Best DMP Trajectory (n_bfs={best_n_bfs})\nRMSE = {rmse_list[best_idx]:.4f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- STEP 5: Plot RMSE vs n_bfs -----
plt.figure(figsize=(6, 4))
plt.plot(n_bfs_list, rmse_list, marker='o', linestyle='-')
plt.title("RMSE vs. Number of Basis Functions")
plt.xlabel("n_bfs")
plt.ylabel("RMSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("SShape_rmse_vs_n_bfs.png", dpi=300, bbox_inches='tight')
plt.show()

# ----- STEP 6: Plot the DMP result for n_bfs = 150 -----
target_n_bfs = 130

if target_n_bfs in n_bfs_list:
    idx = n_bfs_list.index(target_n_bfs)
    y_pred_150 = trajectories[idx]
    rmse_150 = rmse_list[idx]

    plt.figure(figsize=(6, 5))
    plt.plot(y_demo_trimmed[:, 0], y_demo_trimmed[:, 1], 'k--', label='Demonstration')
    plt.plot(y_pred_150[:, 0], y_pred_150[:, 1], 'b-', label=f'DMP (n_bfs={target_n_bfs})')
    plt.scatter(*y_demo[0], c='g', label='Start')
    plt.scatter(*y_demo[-1], c='r', label='Goal')
    plt.title(f"DMP Trajectory (n_bfs={target_n_bfs})\nRMSE = {rmse_150:.4f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("SShape_dmp_trajectory_n_bfs_130.png", dpi=300, bbox_inches='tight')
    plt.show()
else:
    print(f"n_bfs = {target_n_bfs} not found in n_bfs_list")


