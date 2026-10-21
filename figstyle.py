"""Shared figure style for the revised manuscript (static, print figures).

Categorical colours follow a validated fixed order (CVD-checked on the
adjacent pairlist); every series also carries a marker so identity never
relies on colour alone.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100',
         magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948',
         ink='#0b0b0b', ink2='#52514e', grid='#e4e3df', surface='#ffffff')

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.6, 'axes.edgecolor': C['ink2'],
    'axes.labelcolor': C['ink'], 'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
    'grid.color': C['grid'], 'grid.linewidth': 0.6, 'lines.linewidth': 1.4,
    'savefig.dpi': 300, 'figure.dpi': 150, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'dejavusans'})

# correctors -------------------------------------------------------------------
MSHORT = dict(BA='Bordered + line search', BA0='Bordered, no line search', SDC='Standard DC (half period)',
              BAP='Bordered, projected start', ORIG='Original method', CHORD='Chord', FP='Fixed point')
MCOL = dict(BA=C['blue'], BA0=C['orange'], SDC=C['aqua'], BAP=C['yellow'], ORIG=C['magenta'],
            CHORD=C['green'], FP=C['violet'])
MMARK = dict(BA='o', BA0='s', SDC='^', BAP='v', ORIG='D', CHORD='P', FP='X')

# families / systems ----------------------------------------------------------------
FAMORDER = ['lyapunov', 'halo', 'dro']
FAMNAME = dict(lyapunov='$L_1$ Lyapunov', halo='$L_2$ Halo', dro='DRO')
FAMCOL = dict(lyapunov=C['blue'], halo=C['orange'], dro=C['aqua'])
SYSORDER = ['earth-moon', 'saturn-titan', 'sun-earth', 'mars-phobos']
SYSNAME = {'earth-moon': 'Earth-Moon', 'saturn-titan': 'Saturn-Titan', 'sun-earth': 'Sun-Earth',
           'mars-phobos': 'Mars-Phobos'}
SYSMARK = {'earth-moon': 'o', 'saturn-titan': 's', 'sun-earth': '^', 'mars-phobos': 'D'}
