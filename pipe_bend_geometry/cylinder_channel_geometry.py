import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim



def make_grid(x_length=10, y_length=5, nr_xs=100, nr_ys=50, make_plot=False):
    """Make a rectangular grid, with coordinates centered on (0, 0)."""
    xs = np.linspace(0, x_length, num=nr_xs) - x_length/2
    ys = np.linspace(0, y_length, num=nr_ys) - y_length/2
    X, Y = np.meshgrid(xs, ys)

    if make_plot:
        plt.scatter(X, Y)
        plt.axis('equal')
        plt.show()

    return X, Y


def sample_channel_boundary(grid, make_plot=False):
    """Returns the four sides of the grid corresponding to inlet, outlet and channel walls."""
    xs = grid[0]
    ys = grid[1]

    # Extract inlet coordinates.
    inlet_x = xs[0, 0]
    inlet_ys = ys[:, 0]
    inlet_xys = np.zeros((inlet_ys.shape[0], 2))
    inlet_xys[:, 0] = inlet_x
    inlet_xys[:, 1] = inlet_ys

    # Extract outlet coordinates.
    outlet_x = xs[0, -1]
    outlet_ys = ys[:, -1]
    outlet_xys = np.zeros((outlet_ys.shape[0], 2))
    outlet_xys[:, 0] = outlet_x
    outlet_xys[:, 1] = outlet_ys

    # Extract channel left wall.
    left_wall_xs = xs[-1, :]
    left_wall_y = ys[-1, 0]
    left_wall_xys = np.zeros((left_wall_xs.shape[0], 2))
    left_wall_xys[:, 0] = left_wall_xs
    left_wall_xys[:, 1] = left_wall_y

    # Extract channel right wall.
    right_wall_xs = xs[0, :]
    right_wall_y = ys[0, 0]
    right_wall_xys = np.zeros((right_wall_xs.shape[0], 2))
    right_wall_xys[:, 0] = right_wall_xs
    right_wall_xys[:, 1] = right_wall_y

    if make_plot:
        plt.scatter(inlet_xys[1:-1, 0], inlet_xys[1:-1, 1]) # Exclude top and bottom points, corresponding to the wall.
        plt.scatter(outlet_xys[1:-1, 0], outlet_xys[1:-1, 1]) # Exclude top and bottom points, corresponding to the wall.
        plt.scatter(left_wall_xys[:, 0], left_wall_xys[:, 1])
        plt.scatter(right_wall_xys[:, 0], right_wall_xys[:, 1])
        plt.axis('equal')
        plt.show()

    return inlet_xys, outlet_xys, left_wall_xys, right_wall_xys


def exclude_cylinder_points(grid, cylinder_center=(0, 0), cylinder_radius=1, make_plot=False):
    """Excludes the points in the grid that lie within the cylinder of the given center and radius."""

    xs, ys = grid
    
    norms = np.sqrt((xs - cylinder_center[0])**2 + (ys - cylinder_center[1])**2)

    mask = norms <= cylinder_radius

    # Apply the mask to the original mesh grid
    xs_filtered = np.ma.masked_where(mask, xs)
    ys_filtered = np.ma.masked_where(mask, ys)

    if make_plot:
        plt.figure(figsize=(20, 10))
        plt.scatter(xs_filtered, ys_filtered)
        plt.axis('equal')
        plt.show()

    return xs_filtered, ys_filtered


def sample_cylinder_boundary_and_compute_normals(cylinder_center, cylinder_radius, nr_pts, make_plot=False):
    """Samples points on a circle with the given center and radius. Computes normals pointing away from center."""
    
    thetas = np.linspace(0, 2*np.pi, nr_pts)

    xs = cylinder_center[0] + cylinder_radius * np.cos(thetas)
    ys = cylinder_center[1] + cylinder_radius * np.sin(thetas)

    dx_dtheta = cylinder_radius * np.sin(thetas)
    dy_dtheta = -cylinder_radius * np.cos(thetas)
    normal_vectors = np.array([(-dy_dtheta[i], dx_dtheta[i]) for i in range(nr_pts)])

    # Translate the normal vectors to the correct position. 
    normal_vectors[:, 0] = normal_vectors[:, 0] + xs
    normal_vectors[:, 1] = normal_vectors[:, 1] + ys

    # Normalize the vectors to have norm 1. 
    norms = np.linalg.norm(normal_vectors, axis=1)
    normal_vectors = normal_vectors / norms[:, np.newaxis]

    if make_plot:
        plt.scatter(xs, ys)
        plt.quiver(xs, ys, normal_vectors[:, 0], normal_vectors[:, 1], color='red', scale=5, scale_units='xy', angles='xy', label='Normal Vectors')
        plt.axis('equal')
        plt.show()

    return xs, ys, normal_vectors


