import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -----------------------------
# Define Parametric Shapes for the Wormhole Model
# -----------------------------
# Parameter grid (U, V):
#   - U in [0,1] acts as the radial parameter (for the disk or dimple) or vertical parameter (for the tube)
#   - V in [0,2c] is the angular parameter.
u = np.linspace(0, 1, 50)
v = np.linspace(0, 2 * np.pi, 50)
U, V = np.meshgrid(u, v)

def entry_region(U, V):
    """
    Entry Region as a flat disk (plane).
    The disk has radius 1 and lies in the z = 0 plane.
    """
    x = U * np.cos(V)
    y = U * np.sin(V)
    z = np.zeros_like(U)
    return x, y, z

def throat_region(U, V):
    """
    Wormhole Throat: Modeled as a cylindrical tube.
    The tube has a fixed radius of 1 and spans z from -1 to 1.
    """
    x = np.cos(V)
    y = np.sin(V)
    z = 2 * U - 1  # maps U from [0,1] to z in [-1,1]
    return x, y, z

def exit_region(U, V):
    """
    Exit Region: Modeled as a bent tube ("macaroni" shape).
    Represented as a segment of a torus.
    """
    R0 = 1.0  # Major (centerline) radius (controls bending)
    r  = 0.3  # Minor (tube) radius
    theta = np.pi * U  # bending angle from 0 to c (half-torus)
    x = (R0 + r * np.cos(V)) * np.cos(theta)
    y = (R0 + r * np.cos(V)) * np.sin(theta)
    z = r * np.sin(V)
    return x, y, z

def healing_region(U, V):
    """
    Healing Region: Represents a dimple in spacetime.
    The tube dissipates leaving a shallow depression.
    Here we model a parabolic dimple: the disk is no longer flat;
    its center is depressed (z negative) while the boundary remains near z = 0.
    """
    d = 0.3  # Depth of the dimple
    x = U * np.cos(V)
    y = U * np.sin(V)
    z = -d * (1 - U**2)  # Maximum depression at the center (U=0) and flat at the boundary (U=1)
    return x, y, z

# -----------------------------
# Define a Spacetime Field Function
# -----------------------------
def spacetime_field(X, Y, t_current):
    """
    A sample spacetime field representing oscillatory energy density.
    Defined over a 2D grid; oscillates in time.
    """
    return 0.5 * np.cos(2 * np.pi * 0.5 * t_current) * np.exp(-0.5 * (X**2 + Y**2))

# Create a grid for the spacetime field (covers a larger region than the wormhole model)
X_space, Y_space = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))

# -----------------------------
# Set Up Simulation Parameters and Figure
# -----------------------------
T = 10         # Total simulation time (for computing t_current)
dt = 0.01      # Time step in seconds
t = np.arange(0, T, dt)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

def update(frame):
    global dt
    # Extend morphing parameter p to cover a full cycle from 0 to 4.
    # Using 400 frames so that p = frame / 100.
    p = (frame % 400) / 100.0  # p in [0, 4)
    
    # Compute coordinates for each region.
    x_entry, y_entry, z_entry = entry_region(U, V)
    x_throat, y_throat, z_throat = throat_region(U, V)
    x_exit, y_exit, z_exit = exit_region(U, V)
    x_heal, y_heal, z_heal = healing_region(U, V)
    
    # Determine the morphing phase based on p.
    if p <= 1:
        # Phase 1: Morph from Entry (flat disk) to Throat (tube)
        lam = p  # Interpolation parameter (0 to 1)
        x_w = (1 - lam) * x_entry + lam * x_throat
        y_w = (1 - lam) * y_entry + lam * y_throat
        z_w = (1 - lam) * z_entry + lam * z_throat
        morph_label = "Entry (Plane)  Throat (Tube)"
    elif p <= 2:
        # Phase 2: Morph from Throat (tube) to Exit (bent tube)
        mu = p - 1  # Interpolation parameter (0 to 1)
        x_w = (1 - mu) * x_throat + mu * x_exit
        y_w = (1 - mu) * y_throat + mu * y_exit
        z_w = (1 - mu) * z_throat + mu * z_exit
        morph_label = "Throat (Tube)  Exit (Bent Tube)"
    elif p <= 3:
        # Phase 3: Morph from Exit (bent tube) to Healing (dimple)
        nu = p - 2  # Interpolation parameter (0 to 1)
        x_w = (1 - nu) * x_exit + nu * x_heal
        y_w = (1 - nu) * y_exit + nu * y_heal
        z_w = (1 - nu) * z_exit + nu * z_heal
        morph_label = "Exit (Bent Tube)  Healing (Dimple)"
    else:
        # Phase 4: Morph from Healing (dimple) to normalized Entry (flat disk)
        gamma = p - 3  # Interpolation parameter (0 to 1)
        x_w = (1 - gamma) * x_heal + gamma * x_entry
        y_w = (1 - gamma) * y_heal + gamma * y_entry
        z_w = (1 - gamma) * z_heal + gamma * z_entry
        morph_label = "Healing (Dimple)  Normalization (Flat)"
    
    # Clear the axes for the new frame.
    ax.cla()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Compute current time.
    t_current = frame * dt
    
    # Plot the underlying spacetime field.
    Z_space = spacetime_field(X_space, Y_space, t_current)
    ax.plot_surface(X_space, Y_space, Z_space, cmap='plasma', alpha=0.4, edgecolor='none')
    
    # Plot the wormhole model.
    wormhole_surf = ax.plot_surface(x_w, y_w, z_w, cmap='viridis', edgecolor='none')
    
    # Update the title to reflect the current phase and time.
    ax.set_title(f"Wormhole Model: {morph_label} (p = {p:.2f}, t = {t_current:.2f} s)")
    
    # Use a fixed view angle.
    ax.view_init(elev=30, azim=45)
    
    return wormhole_surf

# Create the animation with 400 frames (update every 50 milliseconds).
ani = animation.FuncAnimation(fig, update, frames=400, interval=50, blit=False)

plt.show()

