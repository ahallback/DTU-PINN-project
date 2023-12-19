import matplotlib.pyplot as plt

def Plot2Dquad(X, Y, *args):
    str = 'k'
    if len(args) > 0:
        str = args[0]
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            plt.plot(X[i, :], Y[i, :], str, linewidth=1)
            plt.plot(X[:, j], Y[:, j], str, linewidth=1)
    
    plt.show()