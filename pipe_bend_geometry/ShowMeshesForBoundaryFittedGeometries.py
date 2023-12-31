import numpy as np
import matplotlib.pyplot as plt
from pipe_bend_geometry.TransfiniteQuadMap import TransfiniteQuadMap
from pipe_bend_geometry.DefineBoundaryFittedGeometries import define_geometry
from pipe_bend_geometry.printme import printme
from pipe_bend_geometry.LegendreGaussLobattoNodesAndWeights import LegendreGaussLobattoNodesAndWeights

def show_mesh_for_boundary_fitted_geometries(geometry, radius = 3, print_on_off=False, plot=False):
    # Set print_on_off to 0 (printOFF)

    # Define the directory
    # dir = 'C:\\Users\\ahal\\DTU Kurser\\Deep Learning\\PINN Projekt\\figures\\'

    # PARAMETERS
    M = 28  # xi
    N = M  # eta
    r = radius

    # COMPUTATIONAL GRID
    xi, _ = LegendreGaussLobattoNodesAndWeights(M-1)
    eta = np.copy(xi)
    XI, ETA = np.meshgrid(xi, eta)

    # GEOMETRY
    gam1, _, gam2, _,  gam3, _, gam4, _ = define_geometry(geometry, r, show_geometry=plot)

    # SETUP PHYSICAL GRID
    # Assuming you have already defined gam1, gam2, gam3, gam4
    X, Y = TransfiniteQuadMap(XI, ETA, gam1, gam2, gam3, gam4)  # Physical coordinates

    if plot:
        # Plot the mesh
        plt.figure()
        plt.plot(X, Y, 'k', linewidth=1)
        plt.plot(X.T, Y.T, 'k', linewidth=1)

        if geometry == 'SQ':
            plt.xlabel('r')
            plt.ylabel('s')
        else:
            plt.xlabel('x')
            plt.ylabel('y')

        plt.title('Mesh ' + geometry)
        plt.axis('equal')

        xmin = np.min(X)
        xmax = np.max(X)
        ymin = np.min(Y)
        ymax = np.max(Y)
        Lx = xmax - xmin
        Ly = ymax - ymin
        fac = 0.05

        plt.axis([xmin - fac * Lx, xmax + fac * Lx, ymin - fac * Ly, ymax + fac * Ly])
        plt.show()

    return X, Y

    # filename = 'MeshGeometry' + geometry + '.eps'
    # printme(plt.gcf(), dir, filename, print_on_off)
