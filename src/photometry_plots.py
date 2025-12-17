### Libraries ###
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import Normalize, LogNorm
import math

def plot_cmd(x, y, color_value=None, cmap="gnuplot", xlabel=None, ylabel=None, title=None, 
             cbar_label=None, s=25, alpha=0.9, log_cbar=False, show_nocolor_value=True):

    color = np.asarray(x)
    mag = np.asarray(y)

    plt.figure(figsize=(7, 6))

    if color_value is None:
        # Simple black markers
        plt.scatter(color, mag, s=s, color="black", edgecolor="none")

    else:
        color_value = np.asarray(color_value, dtype=float)

        # Mask
        ok = np.isfinite(color_value)
        bad = ~ok

        # Normalize
        if log_cbar and np.any(ok):
            norm = LogNorm(vmin=np.min(color_value[ok]), vmax=np.max(color_value[ok]))
        else:
            norm = None

        # Good points
        if np.any(ok):
            sc = plt.scatter(
                color[ok], mag[ok], s=s, alpha=alpha,
                c=color_value[ok], cmap=cmap,
                edgecolor="none", norm=norm, zorder=3
            )
       
        if np.any(bad) and show_nocolor_value:
            plt.scatter(
                color[bad], mag[bad], s=s, alpha=alpha,
                facecolor='none', edgecolor='black',
                linewidth=1.0, zorder=1
            )

        # Colorbar
        if np.any(ok):
            cb = plt.colorbar()
            if cbar_label:
                cb.set_label(f"{cbar_label}")

    # Labels
    if xlabel: plt.xlabel(xlabel, fontsize=14)
    if ylabel: plt.ylabel(ylabel, fontsize=14)
    if title:  plt.title(title)

    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ccd(x, y, color_value=None, cmap="gnuplot", xlabel=None, ylabel=None, title=None,
            cbar_label=None, s=25, alpha=0.9, log_cbar=False, show_nocolor_value=True):
    """
    Plot a color-color diagram (CCD).

    Parameters
    ----------
    band1, band2 : array-like
        Bands for x-axis color (band1 - band2)
    band3, band4 : array-like
        Bands for y-axis color (band3 - band4)
    
    color_value : array-like or None
        Optional values for coloring points. NaNs plotted as empty points with black edges.
    
    cmap : str
        Matplotlib colormap name.
    
    s : float
        Marker size.
    
    xlabel, ylabel, title : str
        Axis labels and plot title.
    
    log_cbar : bool
        If True, colorbar is logarithmic.
    """

    x_color = np.asarray(x)
    y_color = np.asarray(y)

    plt.figure(figsize=(7, 6))

    if color_value is None:
        plt.scatter(x_color, y_color, s=s, color="black", edgecolor="none")
    
    else:
        color_value = np.asarray(color_value, dtype=float)
        ok = np.isfinite(color_value)
        bad = ~ok

        # Normalize
        if log_cbar and np.any(ok):
            norm = LogNorm(vmin=np.min(color_value[ok]), vmax=np.max(color_value[ok]))
        else:
            norm = None

        # Good points
        if np.any(ok):
            sc = plt.scatter(x_color[ok], y_color[ok], s=s, alpha=alpha,
                             c=color_value[ok], cmap=cmap,
                             edgecolor="none", norm=norm, zorder=3)
        
        # Missing points
        if np.any(bad) and show_nocolor_value:
            plt.scatter(x_color[bad], y_color[bad], s=s, alpha=alpha,
                        facecolor='none', edgecolor='black', linewidth=1.0, zorder=1)
        
        # Colorbar
        if np.any(ok):
            cb = plt.colorbar(sc)
            if cbar_label:
                cb.set_label(cbar_label)

    # Labels
    if xlabel: plt.xlabel(xlabel, fontsize=14)
    if ylabel: plt.ylabel(ylabel, fontsize=14)
    if title:  plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_histograms(df, columns, bins=30, sharey=False, log=False,
                         base_width=4, base_height=3, max_bins=None):
    """
    Plot histograms for a list of DataFrame columns using an automatically
    computed grid layout (nrows × ncols).

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.

    columns : list of str
        Names of columns to plot.

    bins : int
        Number of histogram bins.

    sharey : bool
        If True, all histograms share the same y-axis.

    base_width : float
        Width of each subplot.

    base_height : float
        Height of each subplot.
    """

    # Filter numeric columns only
    valid_cols = []
    for col in columns:
        if col not in df.columns:
            print(f"[Warning] Column '{col}' not found, skipping.")
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"[Warning] Column '{col}' is not numeric, skipping.")
            continue
        valid_cols.append(col)

    n = len(valid_cols)
    if n == 0:
        raise ValueError("No valid numeric columns to plot.")

    # Automatically determine grid size
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    # Create figure with adaptive size
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(base_width * ncols, base_height * nrows),
        sharey=sharey
    )

    # axes is a 2D array; flatten for easy looping
    axes = np.array(axes).reshape(-1)

    # Plot histograms
    for ax, col in zip(axes, valid_cols):
        if max_bins and max(df[col].dropna())>max_bins:
            ax.hist(df[col].dropna(), bins=np.arange(0,max_bins,(max_bins)/bins), log=log)
        else:
            ax.hist(df[col].dropna(), bins=bins, log=log)
        # ax.set_title(col)
        ax.set_xlabel(col)
        # ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    # Empty remaining axes if grid > number of plots
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


