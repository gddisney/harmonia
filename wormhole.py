import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is enabled
import matplotlib.animation as animation

# -----------------------------
# Simulation Setup and Data Generation
# -----------------------------

# Simulation parameters
T = 10       # Total simulation time in seconds
dt = 0.01    # Time step in seconds
t = np.arange(0, T, dt)
n_time = len(t)
n_space = 100  # Number of spatial points along the Y dimension
Y = np.linspace(-1, 1, n_space)

# Generate synthetic 3D curvature evolution data.
# Replace this synthetic data with your computed curvature_3D array if available.
# Here we simulate a curvature that oscillates with time (using cosine) and is modulated spatially by a Gaussian.
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
c = 3.0e8        # Speed of light (m/s)
curvature_factor = 8 * np.pi * G / c**4

# Create a meshgrid for time and the spatial dimension Y
X_time, Y_space = np.meshgrid(t, Y)
# Synthetic curvature: shape (n_space, n_time)
curvature_3D = curvature_factor * np.cos(2 * np.pi * t) * np.exp(-Y_space**2)

# Compute additional (computed) data:
# Mean curvature over the spatial dimension (for each time step)
mean_curvature = np.mean(curvature_3D, axis=0)
# Estimated wormhole throat radius (example formula)
rho = 1e-27  # Average energy density in kg/m^3
r_throat = np.sqrt(np.abs(mean_curvature) / (2 * np.pi * G * rho))

# -----------------------------
# Animation Setup
# -----------------------------

# Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a persistent annotation text object on the figure (position in figure coordinates)
annotation_text_obj = fig.text(0.05, 0.95, "", fontsize=12,
                               bbox=dict(facecolor='white', alpha=0.8))

# Define the window size (number of time steps to display per frame)
window_size = 200  # Adjust as needed

def update(frame):
    # Clear the axes for the current frame
    ax.cla()
    
    # Determine the time window for the current frame
    start = frame
    end = min(frame + window_size, n_time)
    t_window = t[start:end]
    
    # Create a meshgrid for the current time window (using the same spatial grid Y)
    X_window, Y_window = np.meshgrid(t_window, Y)
    # Extract the corresponding curvature values for this window
    Z_window = curvature_3D[:, start:end]
    
    # Plot the 3D surface for the current window
    surf = ax.plot_surface(X_window, Y_window, Z_window, cmap="coolwarm", edgecolor='none')
    
    # Set axis labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spatial Dimension")
    ax.set_zlabel("Curvature (1/m^2)")  # Changed from 1/m} to 1/m^2
    current_time = t[frame]
    ax.set_title(f"3D Curvature Evolution at t = {current_time:.2f} s")
    
    # Retrieve computed data for the current frame
    current_mean_curvature = mean_curvature[frame]
    current_throat = r_throat[frame]
    
    # Create annotation text with computed data (using 1/m^2 instead of 1/m})
    annotation_text = (f"Time: {current_time:.2f} s\n"
                       f"Mean Curvature: {current_mean_curvature:.2e} 1/m^2\n"
                       f"Throat Radius: {current_throat:.2e} m")
    # Update the persistent annotation text object
    annotation_text_obj.set_text(annotation_text)
    
    # Set a fixed view angle (remove rotation)
    ax.view_init(elev=30, azim=45)
    
    return surf,

# Define frames (here, we step through time in increments of 10 time steps)
frames = np.arange(0, n_time - window_size, 10)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

# To display the animation inline in a Jupyter Notebook, uncomment the following lines:
# from IPython.display import HTML
# HTML(ani.to_jshtml())

plt.show()

