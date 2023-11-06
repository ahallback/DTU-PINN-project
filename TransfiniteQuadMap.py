import numpy as np

def TransfiniteQuadMap(xi, eta, gam1, gam2, gam3, gam4):
    # Initialize arrays for x and y
    x = np.zeros_like(xi)
    y = np.zeros_like(eta)
    
    xi = np.copy(xi)
    eta = np.copy(eta)
    
    # Vertice coordinates of quadrilateral
    tmp = gam1(-1)
    x1 = tmp[0]
    y1 = tmp[1]
    
    tmp = gam1(1)
    x2 = tmp[0]
    y2 = tmp[1]
    
    tmp = gam3(1)
    x3 = tmp[0]
    y3 = tmp[1]
    
    tmp = gam3(-1)
    x4 = tmp[0]
    y4 = tmp[1]
    
    # Interpolated coordinates of physical boundaries
    X1, Y1 = gam1(xi)
    X2, Y2 = gam2(eta)
    X3, Y3 = gam3(xi)
    X4, Y4 = gam4(eta)
    
    # Bilinear mapping from reference square to straight-sided quadrilateral
    h1 = 1 - xi
    h2 = 1 - eta
    h3 = 1 + xi
    h4 = 1 + eta
    
    x = 0.5 * (h1 * X4 + h3 * X2 + h2 * X1 + h4 * X3) - 0.25 * (h1 * (h2 * x1 + h4 * x4) + h3 * (h2 * x2 + h4 * x3))
    y = 0.5 * (h1 * Y4 + h3 * Y2 + h2 * Y1 + h4 * Y3) - 0.25 * (h1 * (h2 * y1 + h4 * y4) + h3 * (h2 * y2 + h4 * y3))
    
    return x, y

#, X1, Y1, X2, Y2, X3, Y3, X4, Y4
