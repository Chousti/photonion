# Copyright (C) 2024 Richard Stiskalek, Nicholas Choustikov
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Various plotting utilities.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_pit(ax, pit):
    """
    Plot P-P plot of PIT values on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    pit : 1-dimensional array
        The PIT values to plot.

    Returns
    -------
    None
    """
    unicov = [np.sort(np.random.uniform(0, 1, len(pit))) for j in range(1000)]
    unip = np.percentile(unicov, [5, 16, 84, 95], axis=0)

    cdf = np.linspace(0, 1, len(pit))
    plt.plot(pit, cdf)
    plt.fill_between(cdf, unip[0], unip[-1], color='gray', alpha=0.2)
    plt.fill_between(cdf, unip[1], unip[-2], color='gray', alpha=0.4)

    plt.plot(cdf, cdf, "k--")

    ax.set_xlabel("Predicted percentile")
    ax.set_ylabel("Empirical percentile")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
