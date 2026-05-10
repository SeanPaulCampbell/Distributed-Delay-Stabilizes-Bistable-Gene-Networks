import os
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import Functions_Gillespie as Gill
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}\DeclareMathOperator{\E}{\mathbb{E}}\DeclareMathOperator{\cv}{cv}')

mean = 3
mu_range = [mean] 
cv_range = [thing.item() for thing in np.round(np.linspace(0,.6,31),2)]
cvs = len(cv_range)
log_rate_range = np.linspace(0,8,17)
rate_range = [thing.item() for thing in 2**log_rate_range]
rates = len(rate_range)

folder = "2025-3-18/"
system = "CR{}S_31cv_log2".format(mean)
residency_counts = np.load(folder+system+"residency_counts.npy")
mean_residencies = np.load(folder+system+"mean_residencies.npy")
cv_residencies = np.load(folder+system+"cv_residencies.npy")

stdev_residencies = cv_residencies * mean_residencies

mean_residencies = np.flip(mean_residencies, axis=0) / np.mean(mean_residencies[:,0])


small_size = 11
medium_size = 14
bigger_size = 18
plt.rc('font', size=medium_size)          # controls default text sizes
plt.rc('axes', titlesize=medium_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

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

color_map = sns.color_palette("vlag", as_cmap=True)
center_color_value = np.mean(mean_residencies[:,0])
##figure, axes = plt.subplots(2)
##figure.suptitle("Corepressive Toggle with Switching Delays")
##
##sns.heatmap(mean_residencies, ax=axes[0], center=center_color_value, cmap=color_map)#, annot=np.round(mean_residencies/1000))
##axes[0].set_title(r"$\E[R]$")
##axes[0].set_xlabel(r"$\cv[\tau]$")
##axes[0].set_xticklabels([cv_range[index] for index in range(cvs) if index%2==0],rotation=45)
##axes[0].set_ylabel(r"$\log_2(r)$")
##axes[0].set_yticklabels([int(log_rate_range[index]) for index in range(rates) if index%2==0],rotation=45)
##sns.heatmap(cv_residencies, ax=axes[1], vmin=.75, vmax=1.25, cmap=color_map)
##axes[1].set_title(r"$\cv[R]$")
##axes[1].set_xlabel(r"$\cv[\tau]$")
##axes[1].set_xticklabels([cv_range[index] for index in range(cvs) if index%2==0],rotation=45)
##axes[1].set_ylabel(r"$\log_2(r)$")
##axes[1].set_yticklabels([int(log_rate_range[index]) for index in range(rates) if index%2==0],rotation=45)
##plt.tight_layout()
##plt.show()

hm = sns.heatmap(mean_residencies, center=center_color_value, cmap=color_map, vmin=np.min(mean_residencies), vmax=np.max(mean_residencies))
hm.set_title("Corepressive Toggle with Switching Delays")
hm.set_xlabel(r"$\cv[\tau]$")
hm.set_xticklabels([cv_range[index] for index in range(cvs) if index%2==0],rotation=45)
hm.set_ylabel(r"$\log_2(r)$")
hm.set_yticklabels(reversed([round(log_rate_range[index],1) for index in range(rates)]),rotation=45)
ticks = [thing.item() for thing in np.round(np.linspace(np.min(mean_residencies),np.max(mean_residencies),5),4)]
hm.collections[0].colorbar.set_ticks(ticks)
plt.show()
