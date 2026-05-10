import os
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import Functions_Gillespie as Gill
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}\DeclareMathOperator{\E}{\mathbb{E}}\DeclareMathOperator{\cv}{cv}')

mu_range = [2,3,4]
mus = len(mu_range)
cv_range = [thing.item() for thing in np.round(np.linspace(0,1.2,31),2)]
cvs = len(cv_range)
##log_rate_range = np.linspace(0,8,17)
##rate_range = list(2**log_rate_range)
##rates = len(rate_range)

folder = "2025-2-4/"
systems = ["CR{}".format(mu) for mu in mu_range]
mean_residencies = [0 for mu in mu_range]
cv_residencies = [0 for mu in mu_range]
error_bars = [0 for mu in mu_range]
mean0 = [0 for mu in mu_range]
for system in systems:
    index = systems.index(system)
    residency_counts = np.load(folder+system+"residency_count.npy")
    mean_residencies[index] = np.load(folder+system+"mean_residencies.npy")
    cv_residencies[index] = np.load(folder+system+"cv_residencies.npy")
    stdev_residencies = cv_residencies[index] * mean_residencies[index]
    sterr = np.divide(stdev_residencies, np.sqrt(residency_counts))
    mean0[index] = mean_residencies[index][0]
    error_bars[index] = np.array([sterr/mean0[index],sterr/mean0[index]])
    mean_residencies[index] = mean_residencies[index]/mean0[index]

small_size = 12
medium_size = 14
bigger_size = 18
plt.rc('font', size=medium_size)          # controls default text sizes
plt.rc('axes', titlesize=medium_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
##   
####figure, axes = plt.subplots(3,2)
##figure, axes = plt.subplots(2)
##figure.suptitle("Switching Heatmap")
##figure.subplots_adjust(hspace=.5)
##
##axes[0].set_ylabel('R/R0')
##axes[0].set_xlabel('CV[delay]')
##axes[1].set_ylabel('CV of R')
##axes[1].set_xlabel('CV[delay]')
##axes[1].set_ylim(0,2)
##axes[0].legend(['mean 3','mean 4'])
##axes[1].legend(['R0={:.2e}'.format(normalizations[0]),
##                'R0={:.2e}'.format(normalizations[1])])

##color_map = sns.color_palette("icefire", as_cmap=True)
##center_color_value = np.mean(mean_residencies[:,0])
figure, axes = plt.subplots(2)
figure.suptitle("Corepressive Toggle System")
##colors =['darkorchid','royalblue','forestgreen']
##ecolors=['mediumorchid','cornflowerblue','limegreen']
colors = ['darkred','royalblue','forestgreen']
ecolors = ['red','cornflowerblue','limegreen']

for index in range(mus):
    axes[0].errorbar(cv_range, mean_residencies[index], yerr=error_bars[index], fmt='o', capsize=3, ms=1, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
    axes[1].plot(cv_range, cv_residencies[index], color=colors[index])
axes[0].set_title("Residency Mean")
axes[0].set_xlabel(r"$\cv[\tau]$")
axes[0].set_ylabel(r"$\displaystyle{\frac{\E[R]}{\E[R_0]}}$")

axes[1].set_title("Residency CV")
axes[1].set_xlabel(r"$\cv[\tau]$")
axes[1].set_ylabel(r"$\cv[R]$")
axes[1].set_ylim(.5,1.75)
axes[0].legend([r"$\E[\tau]={}$".format(mu) for mu in mu_range])
axes[1].legend([r"$\E[R_0]=1.93\times10^4$",r"$\E[R_0]=7.21\times10^4$",r"$\E[R_0]=2.15\times10^5$"])
plt.tight_layout()
plt.show()

for index in range(mus):
    plt.errorbar(cv_range, mean_residencies[index], yerr=error_bars[index], fmt='o', capsize=3, ms=1, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
plt.legend([r"$\E[\tau]={}$".format(mu) for mu in mu_range])
plt.title(r"Corepressive Toggle")
plt.ylabel(r"$\displaystyle{\frac{\E[R]}{\E[R_0]}}$")
plt.xlabel(r"$\cv[\tau]$")
plt.show()
