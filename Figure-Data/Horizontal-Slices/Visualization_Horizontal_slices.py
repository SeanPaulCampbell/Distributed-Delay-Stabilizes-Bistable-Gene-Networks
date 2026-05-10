import os
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import Functions_Gillespie as Gill
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}\DeclareMathOperator{\E}{\mathbb{E}}\DeclareMathOperator{\cv}{cv}')

mu_range = [3]
mus = len(mu_range)
cv_range = [thing.item() for thing in np.round(np.linspace(0,1,26),2)]
cvs = len(cv_range)
rate_range = [1,16,256]
rates = len(rate_range)
##log_rate_range = np.linspace(0,8,17)
##rate_range = list(2**log_rate_range)
##rates = len(rate_range)

folder = "2025-8-4/"
system = "CR3S_31cv_3slice"
##systems = ["CR{}".format(mu) for mu in mu_range]
##mean_residencies = [0 for mu in mu_range]
##cv_residencies = [0 for mu in mu_range]
##error_bars = [0 for mu in mu_range]
##mean0 = [0 for mu in mu_range]
residency_counts = np.load(folder+system+"residency_counts.npy")
mean_residencies = np.load(folder+system+"mean_residencies.npy")
cv_residencies = np.load(folder+system+"cv_residencies.npy")
attempt_counts = np.load(folder+system+"attempt_counts.npy")
mean_attempt_times = np.load(folder+system+"thresh_means.npy")
cv_attempt_times = np.load(folder+system+"thresh_cvs.npy")
mean_number_of_attempts = np.load(folder+system+"geometric_means.npy")
cv_number_of_attempts = np.load(folder+system+"geometric_cvs.npy")

sterr_residencies = np.divide(mean_residencies * cv_residencies, np.sqrt(residency_counts))
sterr_number_of_attempts = np.divide(mean_number_of_attempts * cv_number_of_attempts, np.sqrt(residency_counts))
sterr_attempt_times = np.divide(mean_attempt_times * cv_attempt_times, np.sqrt(attempt_counts))


''' Normalizations and removal of last three points '''
mean_residencies = mean_residencies[:,:-5]
mean_attempt_times = mean_attempt_times[:,:-5]
mean_number_of_attempts = mean_number_of_attempts[:,:-5]
sterr_residencies = sterr_residencies[:,:-5]
sterr_attempt_times = sterr_attempt_times[:,:-5]
sterr_number_of_attempts = sterr_number_of_attempts[:,:-5]

residency0 = list(mean_residencies[:,0])
attempt_time0 = list(mean_attempt_times[:,0])
attempt_num0 = list(mean_number_of_attempts[:,0])
for index in range(rates):
    mean_residencies[index] = mean_residencies[index] / residency0[index]
    mean_attempt_times[index] = mean_attempt_times[index] / attempt_time0[index]
    mean_number_of_attempts[index] = mean_number_of_attempts[index] / attempt_num0[index]
    sterr_residencies[index] = sterr_residencies[index] / residency0[index]
    sterr_attempt_times[index] = sterr_attempt_times[index] / attempt_time0[index]
    sterr_number_of_attempts[index] = sterr_number_of_attempts[index] / attempt_num0[index]


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

figure, axes = plt.subplots(3)

colors = ["royalblue","black","darkred"]
ecolors = ["cornflowerblue","dimgray","red"]

for index in range(rates):
    axes[0].errorbar(cv_range, mean_residencies[index], yerr=sterr_residencies[index], fmt='o', capsize=3, ms=2, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
    axes[1].errorbar(cv_range, mean_number_of_attempts[index], yerr=sterr_number_of_attempts[index], fmt='o', capsize=3, ms=2, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
    axes[2].errorbar(cv_range, mean_attempt_times[index], yerr=sterr_attempt_times[index], fmt='o', capsize=3, ms=2, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])

axes[2].set_xlabel(r"$\cv[\tau]$")
axes[0].set_title(r"Residency Time")
axes[1].set_title(r"Number of Attempts per Residency")
axes[2].set_title(r"Transition Attempt Time")
figure.legend([r"$r=2^0$",r"$r=2^4$",r"$r=2^8$"])
plt.tight_layout()
plt.show()

