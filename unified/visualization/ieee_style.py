"""
unified/visualization/ieee_style.py
IEEE Transactions on Intelligent Transportation Systems — matplotlib style.

Usage:
    from unified.visualization.ieee_style import apply_ieee_style, IEEE_SINGLE_COL
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=IEEE_SINGLE_COL)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------------
# Column widths (inches)
# ---------------------------------------------------------------------------
IEEE_SINGLE_COL  = (3.5,  2.6)    # 1-col figure, standard height
IEEE_DOUBLE_COL  = (7.16, 2.8)    # 2-col figure (full page width)
IEEE_TALL_SINGLE = (3.5,  4.5)    # 1-col, tall (2×1 or 2×2 subplots)
IEEE_TALL_DOUBLE = (7.16, 5.0)    # 2-col, tall (3×2 subplots)
IEEE_WIDE_SINGLE = (3.5,  2.0)    # 1-col, short/wide

# ---------------------------------------------------------------------------
# Colorblind-friendly palette  (Wong 2011 / IBM Design)
# ---------------------------------------------------------------------------
IEEE_COLORS = {
    'blue':    '#0072B2',
    'orange':  '#E69F00',
    'green':   '#009E73',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'sky':     '#56B4E9',
    'yellow':  '#F0E442',
    'black':   '#000000',
    'gray':    '#999999',
}

# Ordered list for cycling
IEEE_COLOR_LIST = [
    IEEE_COLORS['blue'],
    IEEE_COLORS['orange'],
    IEEE_COLORS['green'],
    IEEE_COLORS['red'],
    IEEE_COLORS['purple'],
    IEEE_COLORS['sky'],
    IEEE_COLORS['yellow'],
]


# ---------------------------------------------------------------------------
# Line styles for B&W printing
# ---------------------------------------------------------------------------
IEEE_LINESTYLES = ['-', '--', '-.', ':']


# ---------------------------------------------------------------------------
# Core style function
# ---------------------------------------------------------------------------

def apply_ieee_style():
    """Apply IEEE Transactions rcParams globally.

    Call once at module level in each visualization file.
    """
    mpl.rcParams.update({
        # --- Font -------------------------------------------------------
        'font.family':              'Times New Roman',
        'font.size':                8,
        'axes.titlesize':           9,
        'axes.labelsize':           8,
        'xtick.labelsize':          7,
        'ytick.labelsize':          7,
        'legend.fontsize':          7,
        'legend.framealpha':        0.85,
        'legend.edgecolor':         '0.75',
        'legend.handlelength':      1.5,
        'legend.labelspacing':      0.25,
        'legend.borderpad':         0.4,
        # --- Lines ------------------------------------------------------
        'lines.linewidth':          1.0,
        'lines.markersize':         4,
        'patch.linewidth':          0.6,
        # --- Axes -------------------------------------------------------
        'axes.linewidth':           0.5,
        'axes.grid':                True,
        'axes.spines.top':          False,
        'axes.spines.right':        False,
        'axes.prop_cycle':          mpl.cycler('color', IEEE_COLOR_LIST),
        # --- Grid -------------------------------------------------------
        'grid.linewidth':           0.4,
        'grid.alpha':               0.3,
        'grid.color':               '0.7',
        # --- Ticks ------------------------------------------------------
        'xtick.major.width':        0.5,
        'ytick.major.width':        0.5,
        'xtick.minor.width':        0.4,
        'ytick.minor.width':        0.4,
        'xtick.direction':          'out',
        'ytick.direction':          'out',
        # --- Figure / save ----------------------------------------------
        'figure.dpi':               150,
        'savefig.dpi':              300,
        'savefig.bbox':             'tight',
        'savefig.pad_inches':       0.02,
        # --- Math -------------------------------------------------------
        'mathtext.fontset':         'stix',
        'axes.formatter.use_mathtext': True,
    })


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def label_subplots(axes, x=-0.18, y=1.06, fontsize=9):
    """Add **(a)**, **(b)** ... labels to each axis (IEEE convention).

    Args:
        axes: single Axes, list, or 2-D array of Axes.
        x, y: position in axes-fraction coordinates.
    """
    flat = np.array(axes).flatten()
    for i, ax in enumerate(flat):
        ax.text(x, y, f'({chr(ord("a") + i)})',
                transform=ax.transAxes,
                fontsize=fontsize, fontweight='bold',
                va='top', ha='left')


def shade_phases(ax, time, phase_list, phase_colors: dict, alpha=0.10):
    """Draw background colour bands by simulation phase.

    Args:
        ax: matplotlib Axes.
        time: 1-D array of time values.
        phase_list: list of phase strings, same length as ``time``.
        phase_colors: dict mapping phase name → colour string.
        alpha: transparency of the shading.
    """
    if not len(phase_list):
        return
    cur, t0 = phase_list[0], time[0]
    for i in range(1, len(time)):
        if phase_list[i] != cur or i == len(time) - 1:
            c = phase_colors.get(cur)
            if c:
                ax.axvspan(t0, time[i], alpha=alpha, color=c, lw=0)
            cur = phase_list[i]
            t0  = time[i]


def save_fig(fig, path, dpi=300, **kwargs):
    """Save *fig* to *path* at 300 dpi (print quality).

    Creates parent directories if they do not exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
    kb = os.path.getsize(path) / 1024
    print(f"[IEEE] Saved {os.path.basename(path)} ({kb:.0f} KB)")


__all__ = [
    'apply_ieee_style',
    'label_subplots',
    'shade_phases',
    'save_fig',
    'IEEE_SINGLE_COL',
    'IEEE_DOUBLE_COL',
    'IEEE_TALL_SINGLE',
    'IEEE_TALL_DOUBLE',
    'IEEE_WIDE_SINGLE',
    'IEEE_COLORS',
    'IEEE_COLOR_LIST',
    'IEEE_LINESTYLES',
]
