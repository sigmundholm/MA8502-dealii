import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots

if __name__ == '__main__':
    base = split(split(os.getcwd())[0])[0]
    skip = 1

    rho = -1
    eps = "0.100000"
    names = {-1: "Douglas-Wang", 0: "Streamline Diffusion", 1: "Galerkin Least Squares"}
    for poly_order in [1, 2]:
        full_path = os.path.join(base, f"build/src/streamline_diffusion/errors-o{poly_order}-eps={eps}-rho={rho}.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, :]

        conv_plots(data, head, title=r"$\textrm{" + names[rho] + ": "
                                     r"polynomial order: " + str(poly_order) + "}$"
                                     r", $\epsilon=" + str(float(eps)) + r"$, $\rho=" + str(rho) + "$", latex=True)
    plt.show()
