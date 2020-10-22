import os
from os.path import split
import numpy as np

from utils.plot import conv_plots

if __name__ == '__main__':
    base = split(split(os.getcwd())[0])[0]
    skip = 0
    for poly_order in [2, 3]:
        full_path = os.path.join(base, f"build/src/streamline_diffusion/errors-o{poly_order}-eps=0.100000.csv")

        head = list(map(str.strip, open(full_path).readline().split(",")))
        data = np.genfromtxt(full_path, delimiter=",", skip_header=True)
        data = data[skip:, :]

        conv_plots(data, head, title=r"$\textrm{Streamline Diffusion: polynomial order: " + str(poly_order) + "}$", latex=True)
