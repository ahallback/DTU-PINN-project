from cylinder_channel_geometry import *

def compute_analytical_solution(grid, U, R):
    """Applies analytical solution of cylinder flow to every point in the grid."""
    X, Y = grid
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    phi = U * r * (1 + R**2/r**2) * np.cos(theta) * (r > R)
    phi -= phi.min() # normalizing by minimum value
    v_r = U * (1 - R**2 / r**2) * np.cos(theta) * (r > R)
    v_theta = -U * (1 + R**2 / r**2) * np.sin(theta) * (r > R)
    v_x = np.cos(theta) * v_r - np.sin(theta) * v_theta
    v_y = np.sin(theta) * v_r + np.cos(theta) * v_theta

    return phi, v_x, v_y

def compute_gradient(grid, phi):
    """Numerically computes velocity as gradient of phi on grid."""
    X, Y = grid
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    v, u = np.gradient(phi, dx, dy)
    return u, v

def make_plot(phi_mapped, grid, uv=None):
    """Plots a solution to the potential flow on the grid."""
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 4))
    m = ax1.pcolormesh(grid[0], grid[1], phi_mapped, shading='Gouraud')
    ax1.contour(grid[0], grid[1], phi_mapped, vmin=-1, vmax=16)
    plt.colorbar(m, ax=ax1)
    ax1.axis('equal')
    ax1.set_title("potential")

    if uv is not None:
        u, v = uv
    else:
        u, v = compute_gradient(grid, phi_mapped)

    ax2.quiver(grid[0], grid[1], u, v)
    ax2.set_title("velocity (grad. of potential)");
    ax2.axis('equal')
    ax3.streamplot(grid[0], grid[1], u, v)
    ax3.set_title('stream plot')
    ax3.axis('equal')
    plt.tight_layout()