import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import constants
import scipy
from typing import List, Dict, Callable
from scipy.optimize import minimize
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import copy
import json
from IPython.display import display, Latex

with open('sw_data/jc.txt') as file_jc:
    jc = json.load(file_jc)
with open('sw_data/zz.txt') as file_zz:
    zz = json.load(file_zz)
with open('sw_data/transm_freq.txt') as file_freq:
    transm_freq = json.load(file_freq)
with open('sw_data/zz_sw.txt') as file_zz_sw:
    zz_sw = json.load(file_zz_sw)
with open('sw_data/mu_yz.txt') as file_cr_sw:
    mu_yz = json.load(file_cr_sw)
with open('sw_data/mu_yi.txt') as file_cr_sw:
    mu_yi = json.load(file_cr_sw)
with open('sw_data/mu_yz_sw.txt') as file_cr_sw:
    mu_yz_sw = json.load(file_cr_sw)
with open('sw_data/mu_yi_sw.txt') as file_cr_sw:
    mu_yi_sw = json.load(file_cr_sw)
with open('data_cr_transmon_fluxonium/data_cr_transmon_fluxonium_linear.txt') as file_res:
    result_sw_lin = json.load(file_res)


def plot_zz(
    jc: dict,
    transm_freq,
    zz,
    zz_sw,
    plot_setup={'fs': 20, 'lw': 2.0, 'lw_levels': 3.0, 'ls': 16, 'fsl':18},
    save=False,
    dark=False
):
    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(6, 6))
    colors_dict = {"0": 'darkred', '1': 'red', '2':'coral'}
    for key in jc.keys():
        ax.semilogy(transm_freq, np.array(zz[key])*1e3, linewidth=plot_setup["lw"], \
            color=colors_dict[key], label='$J_C/ 2 \\pi = {}  \, \\mathrm{{MHz}}$'.format(jc[key]*1e3))
        ax.semilogy(transm_freq, np.array(zz_sw[key])*1e3, linewidth=plot_setup["lw"], \
            color=colors_dict[key], linestyle='--')
    ax.set_xlabel('$\\omega_t/2 \\pi \\, (\\mathrm{GHz})$', fontsize=plot_setup["fs"])
    x_ticks = [4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8]
    x_ticks_latex = ['$4.2$', '$4.4$', '$4.6$', '$4.8$', '$5.0$', '$5.2$', '$5.4$', '$5.6$', '$5.8$']
    plt.xticks(x_ticks, x_ticks_latex)
    y_ticks = [0.1, 0.01]
    y_ticks_latex = ['$10^{-1}$', '$10^{-2}$']
    plt.yticks(y_ticks, y_ticks_latex)
    ax.set_ylabel('$\\xi_{ZZ}/2 \\pi \, (\\mathrm{MHz})$', fontsize=plot_setup['fs'])
    ax.tick_params(axis='both', labelsize=plot_setup["ls"])
    plt.legend(loc='best', fontsize=plot_setup["fsl"], bbox_to_anchor=(0.72,0.5))
    if save:
        plt.savefig("zz_fig.svg")
    plt.show()

plot_zz(jc, transm_freq, zz, zz_sw, save=False, dark=False)