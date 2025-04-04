import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.optimize import NonlinearConstraint
from scipy import optimize
from sklearn import metrics


class LinearDS:
    def __init__(self):
        ## Constructor of the LinearDS class
        ## Creates variables for storing demonstrated data and randomly initializes control matrix A
        self.demo_positions = []
        self.demo_velocities = []
        tmp = np.random.rand(1, 2)
        A_matrix = -(np.dot(tmp.T,tmp)+np.eye(2))
        self.init_A_indices = A_matrix

    def motion_model(self, A_matrix, pos):
        # This method calculates the predicted velocities using a linear DS
        # Input:
        #    A_matrix: 2 times 2 array that represents Matrix A
        #    pos: N times 2 array that contains the positions
        # Return:
        #   pred_vel: N times 2 array that contains the velocities predicted from the Linear model vel = A * pos
        
        return np.dot(pos, A_matrix.T)  # Compute velocity using A*pos
    pass

    def lyapunov_constrains(self,A_indices):
        # This function calculates the constraints for stability of a Linear DS
        # Input:
        #    A_indices: a 4-dimensional vector that contains the indices of control Matrix A
        #Return:
        #    eigenvalues: a 2-dimensional vector that contains the eigenvalues of matrix A
        # Hint: Use the eigavals method from the linalg package of numpy
        
        A_matrix = np.reshape(A_indices, (2, 2))
        eigenvalues = np.linalg.eigvals(A_matrix)  # Compute eigenvalues
        return eigenvalues
    pass

    def objective_fun(self, A_indices):
        # This method calculates the prediction error between the demonstrated and the predicted velocities
        # Input:
        #    A_indices: a 4-dimensional vector that contains the indices of control Matrix A
        # Return: Scalar, predicition error between the demonstrated and predicted velocities
        # Hint: use the metrics.mean_squared_error from the sklearn package
        A_matrix = np.reshape(A_indices, (2, 2))  # Reshape the vector to a 2x2 matrix
        pred_vel = self.motion_model(A_matrix, self.demo_positions)  # Get predicted velocities
        if pred_vel is None:
            raise ValueError("motion_model returned None. Check its implementation.")

        mse = metrics.mean_squared_error(self.demo_velocities, pred_vel)  # Compute MSE
        return mse
    pass

    def train_ds(self):
        # Optimize the objective function wrt the constraints using the trust constrain optimization from scipy package
        # Return: The optimal indices of control matrix A

        A_indices = self.init_A_indices.flatten()

        stability_cons = NonlinearConstraint(self.lyapunov_constrains, -1,-1000)
        optimizations_opts = {"disp": True, "maxiter": 10000}
        res = optimize.minimize(self.objective_fun, A_indices, constraints=[stability_cons],options=optimizations_opts, method="trust-constr")

        optimal_A_indices = res.x
        return optimal_A_indices


    def import_demonstration(self, demo_name):
        # Read demonstrations and calculate velocities

        demo = np.loadtxt(demo_name, delimiter=",", dtype=float)
        self.demo_positions = demo.T
        dt = 0.002
        self.demo_velocities = self.getDerivative(self.demo_positions, dt)

    def getDerivative(self, positions, time):
        # Calculate derivative using the Five-point stencil method

        rows_num = positions.shape[0]
        dt = time
        sampling_rate = dt
        der = np.zeros(positions.shape)
        for n in range(0, rows_num - 1):
            if n == 0:
                der[n, :] = np.zeros(positions.shape[1])
            elif n == 1 or n == rows_num - 2:
                der[n, :] = (positions[n, :] - positions[n - 1, :]) / dt
            elif 1 < n < rows_num - 2:
                der[n, :] = (-positions[n + 2, :] + 8 * positions[n + 1, :] - 8 * positions[n - 1, :] + positions[n - 2, :]) / (12 * dt)
        return der


    def plot_ds(self, optimal_A_indices, save_path="ds_plot.png"):
        # Create streamplots of the learned DS
        A_matrix = np.reshape(optimal_A_indices, (2, 2))

        fig, ax = plt.subplots()
        u = np.linspace(-300, 300, 200)
        v = np.linspace(-300, 300, 200)
        uu, vv = np.meshgrid(u, v)
        u_vel = np.empty_like(uu)
        v_vel = np.empty_like(vv)

        for i in range(0, uu.shape[0]):
            for j in range(0, vv.shape[0]):
                lat_pos = np.array([[uu[i, j], vv[i, j]]])
                lat_vel = np.dot(A_matrix, lat_pos.T).T
                u_vel[i, j] = lat_vel[0, 0]
                v_vel[i, j] = lat_vel[0, 1]

        ax.streamplot(uu, vv, u_vel, v_vel, density=1)
        ax.scatter(self.demo_positions[:, 0], self.demo_positions[:, 1], marker='*', c='r', label="Demonstration")
        ax.streamplot(uu, vv, u_vel, v_vel, density=3, color='g', linewidth=3, start_points=np.array([self.demo_positions[0, :]]))
        plt.legend(loc=1)

        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality save
        print(f"Plot saved as {save_path}")

        plt.show()  # Optionally display the plot



if __name__ == "__main__":
    dynamic_system = LinearDS()
    dynamic_system.import_demonstration("Line.csv")
    optimal_params = dynamic_system.train_ds()
    dynamic_system.plot_ds(optimal_params, save_path="C:/Users/Student/Desktop/AdvancedRobotics/WorkShops/WorkShop 10/Learning_Linear_DS/part_a_plots/Line_ds_plot.png")
