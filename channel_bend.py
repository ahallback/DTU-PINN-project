import numpy as np
from DefineBoundaryFittedGeometries import define_geometry
from ShowMeshesForBoundaryFittedGeometries import show_mesh_for_boundary_fitted_geometries
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad, functional
from PIL import Image


def get_normal_vectors(left_wall_fct, right_wall_fct, inlet_fct, outlet_fct, nr_pts=28, show_normals=False):

    # Choose parametrisation domain.
    xi = np.linspace(-1, 1, nr_pts) # Right wall points
    eta = np.linspace(-1, 1, nr_pts) # Left wall points

    # Get points on boundary.
    left_wall_xs, left_wall_ys = left_wall_fct(eta)
    right_wall_xs, right_wall_ys = right_wall_fct(xi)
    outlet_wall_xs, outlet_wall_ys = outlet_fct(eta)
    inlet_wall_xs, inlet_wall_ys = inlet_fct(xi)

    # Calculate the gradient of the boundary function at the chosen point
    delta = 1e-6  # Small perturbation for numerical gradient calculation

    # dxs
    df_dx_left = (left_wall_fct(eta + delta)[0] - left_wall_fct(eta - delta)[0]) / (2 * delta)
    df_dx_right = (right_wall_fct(xi + delta)[0] - right_wall_fct(xi - delta)[0]) / (2 * delta)
    df_dx_outlet = (outlet_fct(eta + delta)[0] - outlet_fct(eta - delta)[0]) / (2 * delta)
    df_dx_inlet = (inlet_fct(xi + delta)[0] - inlet_fct(xi - delta)[0]) / (2 * delta)

    # dys
    df_dy_left = (left_wall_fct(eta + delta)[1] - left_wall_fct(eta - delta)[1]) / (2 * delta)
    df_dy_right = (right_wall_fct(xi + delta)[1] - right_wall_fct(xi - delta)[1]) / (2 * delta)
    df_dy_outlet = (outlet_fct(eta + delta)[1] - outlet_fct(eta - delta)[1]) / (2 * delta)
    df_dy_inlet = (inlet_fct(xi + delta)[1] - inlet_fct(xi - delta)[1]) / (2 * delta)

    # Construct the normal vector
    normal_vectors_left = np.zeros((df_dx_left.shape[0], 2))
    normal_vectors_right = np.zeros((df_dx_right.shape[0], 2))
    normal_vectors_outlet = np.zeros((df_dx_left.shape[0], 2))
    normal_vectors_inlet = np.zeros((df_dx_right.shape[0], 2))

    # Normals x-coordinate.
    normal_vectors_left[:, 0] = -df_dy_left[:]
    normal_vectors_right[:, 0] = df_dy_right[:]
    normal_vectors_outlet[:, 0] = df_dy_outlet[:]
    normal_vectors_inlet[:, 0] = df_dy_inlet[:]

    # Normals y-coordinate.
    normal_vectors_left[:, 1] = df_dx_left[:]
    normal_vectors_right[:, 1] = -df_dx_right[:]
    normal_vectors_outlet[:, 1] = -df_dx_outlet[:]
    normal_vectors_inlet[:, 1] = -df_dx_inlet[:]

    # Normalize the vectors.
    norms_left = np.linalg.norm(normal_vectors_left, axis=1)
    norms_right = np.linalg.norm(normal_vectors_right, axis=1)
    norms_outlet = np.linalg.norm(normal_vectors_outlet, axis=1)
    norms_inlet= np.linalg.norm(normal_vectors_inlet, axis=1)

    normal_vectors_left = normal_vectors_left / norms_left[:, np.newaxis]
    normal_vectors_right = normal_vectors_right / norms_right[:, np.newaxis]
    normal_vectors_outlet = normal_vectors_outlet / norms_outlet[:, np.newaxis]
    normal_vectors_inlet = normal_vectors_inlet / norms_inlet[:, np.newaxis]

    if show_normals:
        # Plot the walls and the normal vectors
        plt.figure(figsize=(15, 8))

        plt.plot(left_wall_xs, left_wall_ys, label='Left wall')
        plt.plot(right_wall_xs, right_wall_ys, label='Right wall')
        plt.scatter(left_wall_xs, left_wall_ys, color='red', label='Points on left wall')
        plt.scatter(right_wall_xs, right_wall_ys, color='red', label='Points on right wall')
        plt.quiver(left_wall_xs, left_wall_ys, normal_vectors_left[:, 0], normal_vectors_left[:, 1], color='green', scale=100, label='Normal vector left wall')
        plt.quiver(right_wall_xs, right_wall_ys, normal_vectors_right[:, 0], normal_vectors_right[:, 1], color='blue', scale=100, label='Normal vector right wall')

        plt.plot(outlet_wall_xs, outlet_wall_ys, label='outlet')
        plt.plot(inlet_wall_xs, inlet_wall_ys, label='inlet')
        plt.scatter(outlet_wall_xs, outlet_wall_ys, color='red', label='Points outlet')
        plt.scatter(inlet_wall_xs, inlet_wall_ys, color='red', label='Points inlet')

        plt.quiver(outlet_wall_xs, outlet_wall_ys, normal_vectors_outlet[:, 0], normal_vectors_outlet[:, 1], color='green', scale=100, label='Normal vector outlet')
        plt.quiver(inlet_wall_xs, inlet_wall_ys, normal_vectors_inlet[:, 0], normal_vectors_inlet[:, 1], color='blue', scale=100, label='Normal vector inlet')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.axis('equal')
        plt.show()

    return normal_vectors_left, normal_vectors_right, normal_vectors_inlet, normal_vectors_outlet


def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad

def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0])
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def laplace(grad_y, x):
    grad = grad_y
    return divergence(grad, x)

def curl(grad_y,x):
    curl = 0
    grad = grad_y
    for i in range(grad.shape[-1]):
        curl += (-1)**i * torch.autograd.grad(grad[..., i], x, torch.ones_like(grad[..., i]), create_graph=True)[0][..., 1-i:2-i]
    return curl



