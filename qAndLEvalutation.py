import numpy as np

def qAndLEvalutation(x, N):
    # Algorithm 24 from Kopriva (2007)
    # Implemented by Allan P. Engsig-Karup

    x = np.asarray(x)

    if N == 0:
        Ln = x * 0 + 1
        GradLn = 0
        return Ln, GradLn, Ln

    if N == 1:
        Ln = x
        GradLn = 1
        return Ln, GradLn, Ln

    Lnm1 = x * 0 + 1
    Ln = x
    
    GradLnm1 = 0 * x
    GradLn = 0 * x + 1

    for k in range(2, N + 1):
        Lnm2=Lnm1 
        Lnm1=Ln
        GradLnm2 = GradLnm1, 
        GradLnm1 = GradLn

        Ln = ((2 * k - 1) / k) * x * Lnm1 - ((k - 1) / k) * Lnm2
        GradLn = GradLnm2 + (2 * k - 1) * Lnm1

    q = Ln - Lnm2
    Gradq = GradLn - GradLnm2
    Ln = Lnm1

    return q, Gradq, Ln
