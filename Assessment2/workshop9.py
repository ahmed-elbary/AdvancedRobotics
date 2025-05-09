import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # type: ignore
import pydmps
import pydmps.dmp_discrete
from pydmps.dmp_discrete import DMPs_discrete



#check the explaination provided in internet
############ Load a demonstration trajectory as numpy array from the demos folder  ############
# Load the trajectory from CSV into a NumPy array
y_demo = np.loadtxt("C:/Users/Student/Desktop/AdvancedRobotics/Assessment2/datasets/Cshape.csv", delimiter=",")

# Print the shape and a few values to verify
print("Trajectory shape:", y_demo.shape)
print("First few points:\n", y_demo[:5])


############ Create the DMP object and set number of DMPs, RBFs and Starting state ############


# Set parameters for the DMP
n_dmps = 2        # Number of DMPs (2 for a 2D trajectory: x and y)
n_bfs =  50         # Number of Radial Basis Functions
start_state = y_demo[0]  # Starting state (first point of the demo trajectory)

# Create the DMP object
dmp = DMPs_discrete(n_dmps=n_dmps, n_bfs=n_bfs)

# Set the starting state (y0)
dmp.y0 = start_state

# Optional: Train the DMP with the demonstration trajectory
dmp.imitate_path(y_des=y_demo)

# Generate the rollout trajectory
y_pred, dy_pred, ddy_pred = dmp.rollout()

# Print to verify
print("Starting state set to:", dmp.y0)
print("Rollout trajectory shape:", y_pred.shape)


############ Train the weights of the DMPs using the imitate path method ############

forcing_term = dmp.imitate_path(y_des=y_demo)
print("Forcing term shape from imitate_path():", forcing_term.shape)
print("Weights shape:", dmp.w.shape)  # Weights are stored in dmp.w


############ generate a trajectory from the trainned DMP using the rollout method ############

try:
    y_pred, dy_pred, ddy_pred, f_learn = dmp.rollout()
    print("Rollout successful with 4 outputs:")
    print("Positions shape:", y_pred.shape)
except ValueError:
    # Handle case where rollout returns 3 values
    y_pred, dy_pred, ddy_pred = dmp.rollout()
    print("Rollout returned 3 outputs:")
    print("Positions shape:", y_pred.shape)

############ Use Matplotlib to generate the desired plots ############
plt.plot(y_demo[:, 0], y_demo[:, 1], "k--", label="Demonstration")
plt.plot(y_pred[:, 0], y_pred[:, 1], "b-", label="Predicted")
plt.legend()
plt.title("Demonstration vs Predicted Trajectory")
plt.show()
