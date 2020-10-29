import os
from os.path import split
import numpy as np

import matplotlib.pyplot as plt
from utils.plot import conv_plots

if __name__ == '__main__':
    base = split(split(os.getcwd())[0])[0]
    for poly_order in [1, 2]:
        full_path = os.path.join(base, f"build/src/poisson/errors-o{poly_order}-dg.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)

        conv_plots(data, head, title=r"$\textrm{Poisson: polynomial order: " + str(poly_order) + "}$",
                   latex=True)
    plt.show()
