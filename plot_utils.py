import matplotlib.pyplot as plt
import numpy as np


def plot_delta(ax, x, y, alpha=1.0, linefmt='C0-', basefmt='C3-'):
    if alpha is not None:
        minValue = alpha * np.average(np.abs(y))
        px = x[np.abs(y) > minValue]
        py = y[np.abs(y) > minValue]
        print(minValue)
    else:
        px = x
        py = y

    print(px.shape, py.shape)

    ax[0].stem(px, 20*np.log10(np.abs(py)), bottom=20*np.log10(minValue), linefmt=linefmt, basefmt=basefmt)
    ax[1].stem(px, np.angle(py) * 180 / np.pi, bottom=0, linefmt=linefmt, basefmt=basefmt)





