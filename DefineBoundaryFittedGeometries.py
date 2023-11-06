import numpy as np
import matplotlib.pyplot as plt


def define_geometry(geometry, show_geometry=True):
    ex = np.array([1, 0])
    ey = np.array([0, 1])
    
    if geometry == 'M1':
        gam1 = lambda xi: np.array([ex_i * (xi + 2) for ex_i in ex])
        dgam1 = lambda xi: ex * 1
        gam2 = lambda eta: np.array([3 * np.cos(np.pi * (eta + 1) / 4) * ex_i for ex_i in ex]) + np.array([3 * np.sin(np.pi * (eta + 1) / 4) * ey_i for ey_i in ey])
        dgam2 = lambda eta: np.array([(3 * np.pi / 4) * (-1) * np.sin(np.pi * (eta + 1) / 4) * ex_i for ex_i in ex]) + np.array([(3 * np.pi / 4) * np.cos(np.pi * (eta + 1) / 4) * ey_i for ey_i in ey])
        gam3 = lambda xi: np.array([ey_i * (xi + 2) for ey_i in ey])
        dgam3 = lambda xi: ey * 1
        gam4 = lambda eta: np.array([np.cos(np.pi * (eta + 1) / 4) * ex_i for ex_i in ex]) + np.array([np.sin(np.pi * (eta + 1) / 4) * ey_i for ey_i in ey])
        dgam4 = lambda eta: np.array([(np.pi / 4) * (-1) * np.sin(np.pi * (eta + 1) / 4) * ex_i for ex_i in ex]) + np.array([(np.pi / 4) * np.cos(np.pi * (eta + 1) / 4) * ey_i for ey_i in ey])
    elif geometry == 'M2':
        gam1 = lambda xi: np.array([ex_i * (2 + xi) for ex_i in ex])
        dgam1 = lambda xi: ex * 1
        gam2 = lambda eta: np.array([(3 * (1 - eta) / 2) * ex_i + (3 * (1 + eta) / 2) * ey_i for ex_i, ey_i in zip(ex, ey)])
        dgam2 = lambda eta: np.array([(3 * (eta * 0 - 0.5)) * ex_i + (3 * (eta * 0 + 0.5)) * ey_i for ex_i, ey_i in zip(ex, ey)])
        gam3 = lambda xi: np.array([ey_i * (2 + xi) for ey_i in ey])
        dgam3 = lambda xi: ey * 1
        gam4 = lambda eta: np.array([np.cos(np.pi * (eta + 1) / 4) * ex_i + np.sin(np.pi * (eta + 1) / 4) * ey_i for ex_i, ey_i in zip(ex, ey)])
        dgam4 = lambda eta: np.array([(np.pi / 4) * (-1) * np.sin(np.pi * (eta + 1) / 4) * ex_i + (np.pi / 4) * np.cos(np.pi * (eta + 1) / 4) * ey_i for ex_i, ey_i in zip(ex, ey)])
    elif geometry == 'M':
        gam1 = lambda xi: np.array([ex_i * (3 * xi * 0.5) - ey_i * (0.3 + 0.45 * (np.tanh(2 * xi) + 1)) for ex_i, ey_i in zip(ex, ey)])
        dgam1 = lambda xi: np.array([ex_i * (xi * 0 + 1.5) - ey_i * (0.3 + 0.45 * (2 * np.cosh(2 * xi) ** (-2))) for ex_i, ey_i in zip(ex, ey)])
        gam2 = lambda eta: np.array([ex_i * (eta * 0 + 1.5) + ey_i * (eta * (0.3 + 0.45 * (np.tanh(2) + 1))) for ex_i, ey_i in zip(ex, ey)])
        dgam2 = lambda eta: np.array([ex_i * (eta * 0) + ey_i * (eta * 0 + (0.3 + 0.45 * (np.tanh(2) + 1))) for ex_i, ey_i in zip(ex, ey)])
        gam3 = lambda xi: np.array([ex_i * (3 * xi * 0.5) + ey_i * (0.3 + 0.45 * (np.tanh(2 * xi) + 1)) for ex_i, ey_i in zip(ex, ey)])
        dgam3 = lambda xi: np.array([ex_i * (xi * 0 + 1.5) + ey_i * (0.45 * (2 * np.cosh(2 * xi) ** (-2))) for ex_i, ey_i in zip(ex, ey)])
        gam4 = lambda eta: np.array([ex_i * (eta * 0 - 1.5) + ey_i * (eta * (0.3 + 0.45 * (np.tanh(-2) + 1))) for ex_i, ey_i in zip(ex, ey)])
        dgam4 = lambda eta: np.array([ex_i * (eta * 0) + ey_i * (eta * 0 + (0.3 + 0.45 * (np.tanh(-2) + 1))) for ex_i, ey_i in zip(ex, ey)])
    elif geometry == 'M3':
        gam1 = lambda xi: np.array([(ex_i * (8 + 7.75 * np.cos(np.pi + (xi + 1) * np.pi / 4)) + ey_i * (8 + 7.75 * np.cos(np.pi + (xi + 1) * np.pi / 4)) ** -1) for ex_i, ey_i in zip(ex, ey)])
        dgam1 = lambda xi: np.array([(ex_i * ((-np.pi / 4) * 7.75 * np.sin(np.pi + (xi + 1) * np.pi / 4)) + ey_i * ((-np.pi / 4) * 7.75 * np.sin(np.pi + (xi + 1) * np.pi / 4)) ** -1) for ex_i, ey_i in zip(ex, ey)])
        gam2 = lambda eta: np.array([(ex_i * 8 + ey_i * (0.125 + 1.125 * (eta + 1) / 2)) for ex_i, ey_i in zip(ex, ey)])
        dgam2 = lambda eta: np.array([(ex_i * (eta * 0) + ey_i * (eta * 0 + (0.3 + 0.45 * (np.tanh(2) + 1)))) for ex_i, ey_i in zip(ex, ey)])
        gam3 = lambda xi: np.array([(ex_i * (8 + 5.5 * np.cos(np.pi + (xi + 1) / 2 * np.pi / 2)) + ey_i * 10 * ((8 + 5.5 * np.cos(np.pi + (xi + 1) / 2 * np.pi / 2)) ** -1)) for ex_i, ey_i in zip(ex, ey)])
        dgam3 = lambda xi: np.array([(ex_i * (xi * 0 + 3 / 2) + ey_i * (0.45 * (2 * np.sech(2 * xi) ** 2))) for ex_i, ey_i in zip(ex, ey)])
        gam4 = lambda eta: np.array([(ex_i * (0.25 + 2.25 * (eta + 1) / 2) + ey_i * (eta * 0 + 4)) for ex_i, ey_i in zip(ex, ey)])
        dgam4 = lambda eta: np.array([ex_i * (eta * 0) + ey_i * (eta * 0 + (0.3 + 0.45 * (np.tanh(-2) + 1))) for ex_i, ey_i in zip(ex, ey)])
    elif geometry == 'SQ':
        # Geometry for the reference square
        p = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        VX = p[:, 0]
        VY = p[:, 1]
        EToV = np.array([1, 2, 3, 4])
        EToV = EToV - 1
        K = EToV.shape[0]
        Nv = len(VX)
        
        V1 = EToV[0]
        X1 = VX[V1]
        Y1 = VY[V1]
        V2 = EToV[1]
        X2 = VX[V2]
        Y2 = VY[V2]
        V3 = EToV[2]
        X3 = VX[V3]
        Y3 = VY[V3]
        V4 = EToV[3]
        X4 = VX[V4]
        Y4 = VY[V4]
        
        ex=np.array([1, 0])
        ey=np.array([0, 1])

        gam1 = lambda xi: np.array([ex_i * ((xi + 1) * 0.5 * (X2 - X1) + X1) + ey_i * ((xi + 1) * 0.5 * (Y2 - Y1) + Y1) for ex_i, ey_i in zip(ex, ey)])
        dgam1 = lambda xi: np.array([ex_i * (xi * 0 + 0.5 * (X2 - X1)) + ey_i * (xi * 0 + 0.5 * (Y2 - Y1)) for ex_i, ey_i in zip(ex, ey)])
        gam2 = lambda eta: np.array([ex_i * ((eta + 1) * 0.5 * (X3 - X2) + X2) + ey_i * ((eta + 1) * 0.5 * (Y3 - Y2) + Y2) for ex_i, ey_i in zip(ex, ey)])
        dgam2 = lambda eta: np.array([ex_i * (eta * 0 + 0.5 * (X3 - X2)) + ey_i * (eta * 0 + 0.5 * (Y3 - Y2)) for ex_i, ey_i in zip(ex, ey)])
        gam3 = lambda xi: np.array([ex_i * ((xi + 1) * 0.5 * (X3 - X4) + X4) + ey_i * ((xi + 1) * 0.5 * (Y3 - Y4) + Y4) for ex_i, ey_i in zip(ex, ey)])
        dgam3 = lambda xi: np.array([ex_i * (xi * 0 + 0.5 * (X4 - X3)) + ey_i * (xi * 0 + 0.5 * (Y4 - Y3)) for ex_i, ey_i in zip(ex, ey)])
        gam4 = lambda eta: np.array([ex_i * ((eta + 1) * 0.5 * (X4 - X1) + X1) + ey_i * ((eta + 1) * 0.5 * (Y4 - Y1) + Y1) for ex_i, ey_i in zip(ex, ey)])
        dgam4 = lambda eta: np.array([ex_i * (eta * 0 + 0.5 * (X1 - X4)) + ey_i * (eta * 0 + 0.5 * (Y1 - Y4)) for ex_i, ey_i in zip(ex, ey)])
    else:
        # GEOMETRY - REFERENCE SQUARE
        # Define single rectangle
        p = np.array([[2.5, -1], [4, -1], [4.5, 1.3], [2, 1]])  # skewed
        # p = np.array([[2, -1], [4, -1], [4, 1], [2, 1])  # square
        # p = np.array([[2, -1], [5, -1], [5, 1], [2, 1])  # square
        # p = np.array([[2, -1], [4, -1], [4, 2], [2, 2])  # square

        VX = p[:, 0]
        VY = p[:, 1]

        EToV = np.array([1, 2, 3, 4])
        EToV = EToV - 1
        K = EToV.shape[0]
        Nv = len(VX)

        # Kopriva Setup
        V1 = EToV[0]
        X1 = VX[V1]
        Y1 = VY[V1]
        V2 = EToV[1]
        X2 = VX[V2]
        Y2 = VY[V2]
        V3 = EToV[2]
        X3 = VX[V3]
        Y3 = VY[V3]
        V4 = EToV[3]
        X4 = VX[V4]
        Y4 = VY[V4]

        # GEOMETRY - reference square
        ex = np.array([1, 0])
        ey = np.array([0, 1])

        gam1 = lambda xi: np.array([ex_i * ((xi + 1) * 0.5 * (X2 - X1) + X1) + ey_i * ((xi + 1) * 0.5 * (Y2 - Y1) + Y1) for ex_i, ey_i in zip(ex, ey)])
        dgam1 = lambda xi: np.array([ex_i * (0.5 * (X2 - X1)) + ey_i * (0.5 * (Y2 - Y1)) for ex_i, ey_i in zip(ex, ey)])

        gam2 = lambda eta: np.array([ex_i * ((eta + 1) * 0.5 * (X3 - X2) + X2) + ey_i * ((eta + 1) * 0.5 * (Y3 - Y2) + Y2) for ex_i, ey_i in zip(ex, ey)])
        dgam2 = lambda eta: np.array([ex_i * (0.5 * (X3 - X2)) + ey_i * (0.5 * (Y3 - Y2)) for ex_i, ey_i in zip(ex, ey)])

        gam3 = lambda xi: np.array([ex_i * ((xi + 1) * 0.5 * (X3 - X4) + X4) + ey_i * ((xi + 1) * 0.5 * (Y3 - Y4) + Y4) for ex_i, ey_i in zip(ex, ey)])
        dgam3 = lambda xi: np.array([ex_i * (0.5 * (X4 - X3)) + ey_i * (0.5 * (Y4 - Y3)) for ex_i, ey_i in zip(ex, ey)])

        gam4 = lambda eta: np.array([ex_i * ((eta + 1) * 0.5 * (X4 - X1) + X1) + ey_i * ((eta + 1) * 0.5 * (Y4 - Y1) + Y1) for ex_i, ey_i in zip(ex, ey)])
        dgam4 = lambda eta: np.array([ex_i * (0.5 * (X1 - X4)) + ey_i * (0.5 * (Y1 - Y4)) for ex_i, ey_i in zip(ex, ey)])

    if show_geometry:  # Use True to enable this block
        # Show geometry
        xi = np.linspace(-1, 1, 40)
        eta = np.linspace(-1, 1, 40)

        # Define the gam functions as needed
        # gam1 = lambda xi: ex * (3 * xi * 0.5) - ey * (0.4 + 0.4 * (np.tanh(2 * xi) + 1))
        # gam2 = lambda eta: ex * (eta * 0 + 3 / 2) + ey * (eta * (0.4 + 0.4 * (np.tanh(4) + 1)))
        # gam3 = lambda xi: ex * (3 * xi * 0.5) + ey * (0.4 + 0.4 * (np.tanh(2 * xi) + 1))
        # gam4 = lambda eta: ex * (eta * 0 - 3 / 2) + ey * (eta * (0.4 + 0.4 * (np.tanh(-4) + 1)))

        plt.figure()
        plt.title(f'Geometry {geometry}')
        
        tmp = gam1(xi)
        plt.plot(tmp[0, :], tmp[1, :], 'k')
        tmp = gam2(eta)
        plt.plot(tmp[0, :], tmp[1, :], 'k')
        tmp = gam3(xi)
        plt.plot(tmp[0, :], tmp[1, :], 'k')
        tmp = gam4(eta)
        plt.plot(tmp[0, :], tmp[1, :], 'k')

        plt.axis('equal')
        plt.show()

    return gam1, dgam1, gam2, dgam2, gam3, dgam3, gam4, dgam4

## Example usage:
# geometry = 'SQ'
# gam1, dgam1, gam2, dgam2, gam3, dgam3, gam4, dgam4 = define_geometry(geometry)


## Example usage:
# geometry = 'M1'
# gam1, dgam1, gam2, dgam2, gam3, dgam3, gam4, dgam4 = define_geometry(geometry)
