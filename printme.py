import matplotlib.pyplot as plt

def printme(p1, dir, filename, printONOFF):
    if printONOFF:
        plt.savefig(dir + filename, format='eps')