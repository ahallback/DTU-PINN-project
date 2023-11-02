import numpy as np
from qAndLEvalutation import qAndLEvalutation

def LegendreGaussLobattoNodesAndWeights(N):
    x = np.zeros(N + 1)
    w = np.zeros(N + 1)
    nit = 8
    TOL = 4 * np.finfo(float).eps

    if N == 1:
        x[0] = -1
        x[1] = 1
        w[0] = 1
        w[1] = 1
        return x, w

    x[0] = -1
    w[0] = 2 / (N * (N + 1))
    x[N] = 1
    w[N] = w[0]

    for j in range(int(0.5 * (N + 1) - 1)):
        x[j + 1] = -np.cos((j + 1/4) * np.pi / N - 3 / (8 * N * np.pi) / (j + 1/4))

        for k in range(nit):
            q, Gradq, Ln = qAndLEvalutation(x[j + 1], N)
            Delt = -q / Gradq
            x[j + 1] = x[j + 1] + Delt
            if abs(Delt) <= TOL * abs(x[j + 1]):
                break

        q, Gradq, Ln = qAndLEvalutation(x[j + 1], N)
        x[N - j] = -x[j + 1]
        w[j + 1] = 2 / (N * (N + 1) * Ln**2)
        w[N - j] = w[j + 1]

    if N % 2 == 0:
        q, Gradq, Ln = qAndLEvalutation(0, N)
        x[N // 2] = 0
        w[N // 2] = 2 / (N * (N + 1) * Ln**2)

    return x, w