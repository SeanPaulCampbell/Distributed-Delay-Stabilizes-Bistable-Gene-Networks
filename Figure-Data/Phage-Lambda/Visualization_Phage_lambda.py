import os
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import Functions_Gillespie as Gill
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}\DeclareMathOperator{\E}{\mathbb{E}}\DeclareMathOperator{\cv}{cv}')

mu_range = [15,20,25]
mus = len(mu_range)
##cv_range = sorted(list(set([thing.item() for thing in
##                            np.round(np.linspace(0,0.6,31),2)] +
##                           [thing.item() for thing in
##                            np.round(np.linspace(0,0.6,15),2)]  )
##                       )
##                  )
cv_range = [thing.item() for thing in np.round(np.linspace(0,0.6,31),2)]
cvs = len(cv_range)
##log_rate_range = np.linspace(0,8,17)
##rate_range = list(2**log_rate_range)
##rates = len(rate_range)
skip_half_of_15 = False

folder = "2025-10-12/"
system = "PL_gamma"

residency_counts = np.load(folder+system+"_residency_counts.npy")
mean_residencies = np.load(folder+system+"_mean_residencies.npy")
cv_residencies = np.load(folder+system+"_cv_residencies.npy")
attempt_counts = np.load(folder+system+"_attempt_counts.npy")
mean_attempt_times = np.load(folder+system+"_thresh_means.npy")
cv_attempt_times = np.load(folder+system+"_thresh_cvs.npy")
mean_number_of_attempts = np.load(folder+system+"_geometric_means.npy")
cv_number_of_attempts = np.load(folder+system+"_geometric_cvs.npy")

if skip_half_of_15:
    for cv in range(cvs):
        if cv % 2 == 1:
            mean_residencies[0,cv] = np.nan
            cv_residencies[0,cv] = np.nan
            mean_attempt_times[0,cv] = np.nan
            cv_attempt_times[0,cv] = np.nan
            mean_number_of_attempts[0,cv] = np.nan
            cv_number_of_attempts[0,cv] = np.nan

sterr_residencies = np.divide(mean_residencies * cv_residencies, np.sqrt(residency_counts))
sterr_number_of_attempts = np.divide(mean_number_of_attempts * cv_number_of_attempts, np.sqrt(residency_counts))
sterr_attempt_times = np.divide(mean_attempt_times * cv_attempt_times, np.sqrt(attempt_counts))

for mu in range(mus):
    norm = mean_residencies[mu,0]
    mean_residencies[mu,:] = mean_residencies[mu,:] / norm
    sterr_residencies[mu,:] = sterr_residencies[mu,:] / norm

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
figure.suptitle(r"Phage-$\lambda$ System")
##colors =['darkorchid','royalblue','forestgreen']
##ecolors=['mediumorchid','cornflowerblue','limegreen']
colors = ['darkred','royalblue','forestgreen']
ecolors = ['red','cornflowerblue','limegreen']

for index in range(mus):
    axes[0].errorbar(cv_range, mean_residencies[index], yerr=sterr_residencies[index], fmt='o', capsize=3, ms=1, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
    axes[1].plot(cv_range, cv_residencies[index], color=colors[index])
axes[0].set_title(r"Residency Mean")
axes[0].set_xlabel(r"$\cv[\tau]$")
axes[0].set_ylabel(r"$\displaystyle{\frac{\E[R]}{\E[R_0]}}$")

axes[1].set_title(r"Residency CV")
axes[1].set_xlabel(r"$\cv[\tau]$")
axes[1].set_ylabel(r"$\cv[R]$")
axes[1].set_ylim(.5,1.75)
axes[0].legend([r"$\E[\tau]={}$".format(mu) for mu in mu_range])
#axes[1].legend([r"$\E[R_0]=1.93\times10^4$",r"$\E[R_0]=7.21\times10^4$",r"$\E[R_0]=2.15\times10^5$"])
plt.tight_layout()
plt.show()


for index in range(mus):
    plt.errorbar(cv_range, mean_residencies[index], yerr=sterr_residencies[index], fmt='o', capsize=3, ms=3, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
plt.title(r"Phage-$\lambda$ Gamma Distributed")
plt.xlabel(r"$\cv[\tau]$")
plt.ylabel(r"$\displaystyle{\frac{\E[R]}{\E[R_0]}}$")
plt.legend([r"$\E[\tau]={}$".format(mu) for mu in mu_range])
plt.show()


figure, axes = plt.subplots(3)

figure.suptitle(r"Phage-$\lambda$ Gamma Distributed")

for index in range(mus):
    axes[0].errorbar(cv_range, mean_residencies[index], yerr=sterr_residencies[index], fmt='o', capsize=3, ms=2, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
    axes[1].errorbar(cv_range, mean_number_of_attempts[index], yerr=sterr_number_of_attempts[index], fmt='o', capsize=3, ms=2, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
    axes[2].errorbar(cv_range, mean_attempt_times[index], yerr=sterr_attempt_times[index], fmt='o', capsize=3, ms=2, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])

axes[2].set_xlabel(r"$\cv[\tau]$")
axes[0].set_title(r"Residency Time")
axes[1].set_title(r"Number of Attempts per Residency")
axes[2].set_title(r"Transition Attempt Time")
axes[0].set_ylabel(r"$\displaystyle{\frac{\E[R]}{\E[R_0]}}$")
axes[0].legend([r"$\E[\tau]={}$".format(mu) for mu in mu_range])
plt.tight_layout()
plt.show()
