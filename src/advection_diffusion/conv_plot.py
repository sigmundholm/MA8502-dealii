import os
from os.path import split
import numpy as np
import matplotlib.pyplot as plt

from utils.plot import conv_plots

if __name__ == '__main__':
    base = split(split(os.getcwd())[0])[0]
    skip = 1
    eps = "0.001000"
    # eps = "0.100000"
    for poly_order in [1, 2]:
        full_path = os.path.join(base, f"build/src/advection_diffusion/errors-o{poly_order}-eps={eps}-dg.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, :]

        conv_plots(data, head, title=r"$\textrm{Advection Diffusion DG: polynomial order: " + str(poly_order) +
                                     ", eps: " + str(float(eps)) + "}$")

        hs = data[:, 0]
        l2 = data[:, 1]
        h1 = data[:, 2]
        eoc_l2 = np.log(l2[:-1]/l2[1:]) / np.log(hs[:-1]/hs[1:])
        eoc_h1 = np.log(h1[:-1]/h1[1:]) / np.log(hs[:-1]/hs[1:])
        print()
        print("========================================")
        print("EOC (L2): ", eoc_l2)
        print("EOC (H1): ", eoc_h1)
        print("========================================")

    plt.show()
