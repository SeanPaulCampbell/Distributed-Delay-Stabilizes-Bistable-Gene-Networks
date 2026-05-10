import os
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import Functions_Gillespie as Gill
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}\DeclareMathOperator{\E}{\mathbb{E}}\DeclareMathOperator{\cv}{cv}')

mu_range = [3,4,5] 
cv_range = [thing.item() for thing in np.round(np.linspace(0,.6,31),2)]

mean_residencies = np.zeros([2,len(cv_range)]) 
cv_residencies = np.zeros([2,len(cv_range)]) 
mean_high_residencies = np.zeros([2,len(cv_range)]) 
cv_high_residencies = np.zeros([2,len(cv_range)]) 
mean_low_residencies = np.zeros([2,len(cv_range)]) 
cv_low_residencies = np.zeros([2,len(cv_range)]) 

mean_residencies[0,:] = np.load("2024-10-14/mean_residencies.npy")
mean_low_residencies[0,:] = np.load("2024-10-14/mean_low_residencies.npy")
mean_high_residencies[0,:] = np.load("2024-10-14/mean_high_residencies.npy")
cv_residencies[0,:] = np.load("2024-10-14/cv_residencies.npy")
cv_low_residencies[0,:] = np.load("2024-10-14/cv_low_residencies.npy")
cv_high_residencies[0,:] = np.load("2024-10-14/cv_high_residencies.npy")

mean_residencies[1,:] = np.load("2024-09-29/mean_residencies.npy")
mean_low_residencies[1,:] = np.load("2024-09-29/mean_low_residencies.npy")
mean_high_residencies[1,:] = np.load("2024-09-29/mean_high_residencies.npy")
cv_residencies[1,:] = np.load("2024-09-29/cv_residencies.npy")
cv_low_residencies[1,:] = np.load("2024-09-29/cv_low_residencies.npy")
cv_high_residencies[1,:] = np.load("2024-09-29/cv_high_residencies.npy")

err_high = [0,0,0]
err_low = [0,0,0]
err_high[0] = cv_high_residencies[0,:]*mean_high_residencies[0,:]/(mean_high_residencies[0,0]*np.sqrt(5000))
err_low[0] = cv_low_residencies[0,:]*mean_low_residencies[0,:]/(mean_low_residencies[0,0]*np.sqrt(5000))
err_high[1] = cv_high_residencies[1,:]*mean_high_residencies[1,:]/(mean_high_residencies[1,0]*np.sqrt(5000))
err_low[1] = cv_low_residencies[1,:]*mean_low_residencies[1,:]/(mean_low_residencies[1,0]*np.sqrt(5000))

normalizations = [[mean_residencies[ii,0],mean_low_residencies[ii,0],mean_high_residencies[ii,0]] for ii in range(2)]
mean_residencies[0,:] = mean_residencies[0,:] / mean_residencies[0,0]
mean_low_residencies[0,:] = mean_low_residencies[0,:] / mean_low_residencies[0,0]
mean_high_residencies[0,:] = mean_high_residencies[0,:] / mean_high_residencies[0,0]

mean_residencies[1,:] = mean_residencies[1,:] / mean_residencies[1,0]
mean_low_residencies[1,:] = mean_low_residencies[1,:] / mean_low_residencies[1,0]
mean_high_residencies[1,:] = mean_high_residencies[1,:] / mean_high_residencies[1,0]

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
colors = ['darkred','royalblue','forestgreen']
ecolors = ['red','cornflowerblue','limegreen']


##figure, axes = plt.subplots(3,2)
figure, axes = plt.subplots(2,2)
figure.suptitle("Single Species Toggle")

for index in range(2):
##    axes[0,0].plot(cv_range, mean_residencies[index,:]) 
##    axes[0,1].plot(cv_range, cv_residencies[index,:]) 
##    axes[1,0].plot(cv_range, mean_high_residencies[index,:]) 
##    axes[1,1].plot(cv_range, cv_high_residencies[index,:]) 
##    axes[2,0].plot(cv_range, mean_low_residencies[index,:]) 
##    axes[2,1].plot(cv_range, cv_low_residencies[index,:])
    axes[0,0].errorbar(cv_range, mean_high_residencies[index,:], yerr=err_high[index], fmt='o', capsize=3, ms=1, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
    axes[1,0].plot(cv_range, cv_high_residencies[index,:], color=colors[index])
    axes[0,1].errorbar(cv_range, mean_low_residencies[index,:], yerr=err_low[index], fmt='o', capsize=3, ms=1, ecolor=ecolors[index], mfc=colors[index], mec=colors[index])
    axes[1,1].plot(cv_range, cv_low_residencies[index,:], color=colors[index])

cv_range = list(np.round(np.linspace(0,.6,61),2))

mean_residencies = np.zeros([1,len(cv_range)]) 
cv_residencies = np.zeros([1,len(cv_range)]) 
mean_high_residencies = np.zeros([1,len(cv_range)]) 
cv_high_residencies = np.zeros([1,len(cv_range)]) 
mean_low_residencies = np.zeros([1,len(cv_range)]) 
cv_low_residencies = np.zeros([1,len(cv_range)])

mean_residencies[0,:] = np.load("2024-10-23/mean_residencies.npy")
mean_low_residencies[0,:] = np.load("2024-10-23/mean_low_residencies.npy")
mean_high_residencies[0,:] = np.load("2024-10-23/mean_high_residencies.npy")
cv_residencies[0,:] = np.load("2024-10-23/cv_residencies.npy")
cv_low_residencies[0,:] = np.load("2024-10-23/cv_low_residencies.npy")
cv_high_residencies[0,:] = np.load("2024-10-23/cv_high_residencies.npy")
err_high[2] = cv_high_residencies[0,:]*mean_high_residencies[0,:]/(mean_high_residencies[0,0]*np.sqrt(5000))
err_low[2] = cv_low_residencies[0,:]*mean_low_residencies[0,:]/(mean_low_residencies[0,0]*np.sqrt(5000))

##for index in range(len(cv_range)):
##    cv_residencies[0,index] = mean_residencies[0,index] * cv_residencies[0,index]
normalizations.append([mean_residencies[0,0],mean_low_residencies[0,0],mean_high_residencies[0,0]])
mean_residencies[0,:] = mean_residencies[0,:] / mean_residencies[0,0]
mean_low_residencies[0,:] = mean_low_residencies[0,:] / mean_low_residencies[0,0]
mean_high_residencies[0,:] = mean_high_residencies[0,:] / mean_high_residencies[0,0]

##axes[0,0].plot(cv_range, mean_residencies[0,:]) 
##axes[0,1].plot(cv_range, cv_residencies[0,:]) 
##axes[1,0].plot(cv_range, mean_high_residencies[0,:]) 
##axes[1,1].plot(cv_range, cv_high_residencies[0,:]) 
##axes[2,0].plot(cv_range, mean_low_residencies[0,:]) 
##axes[2,1].plot(cv_range, cv_low_residencies[0,:])
axes[0,0].errorbar(cv_range, mean_high_residencies[0,:], yerr=err_high[2], fmt='o', capsize=2, ms=1, ecolor=ecolors[2], mfc=colors[2], mec=colors[2]) 
axes[1,0].plot(cv_range, cv_high_residencies[0,:], color=colors[2]) 
axes[0,1].errorbar(cv_range, mean_low_residencies[0,:], yerr=err_low[2], fmt='o', capsize=2, ms=1, ecolor=ecolors[2], mfc=colors[2], mec=colors[2]) 
axes[1,1].plot(cv_range, cv_low_residencies[0,:], color=colors[2])

figure.subplots_adjust(hspace=.5)

##axes[0,0].set_title('Mean Residency Time')
##axes[0,0].set_ylabel('R/R0')
##axes[0,1].set_title('Residency Time CV')
##axes[0,1].set_ylabel('CV of R')
##axes[1,0].set_title('High State Mean Residency Time')
##axes[1,0].set_ylabel('R/R0')
##axes[1,1].set_title('High State Residency Time CV')
##axes[1,1].set_ylabel('CV of R')
##axes[2,0].set_title('Low State Mean Residency Time')
##axes[2,0].set_ylabel('R/R0')
##axes[2,0].set_xlabel('CV tau')
##axes[2,1].set_title('Low State Residency Time CV')
##axes[2,1].set_ylabel('CV of R')
##axes[2,1].set_xlabel('CV tau')
##axes[2,1].legend(['mean 3','mean 4', 'mean 5'])
axes[0,0].set_title('High State Mean Residency Time')
axes[0,0].set_ylabel(r"$\displaystyle{\frac{\E[R]}{\E[R_0]}}$")
axes[1,0].set_title('High State Residency Time CV')
axes[1,0].set_ylabel(r"$\cv[R]$")
axes[0,1].set_title('Low State Mean Residency Time')
axes[1,0].set_xlabel(r"$\cv[\tau]$")
axes[1,1].set_title('Low State Residency Time CV')
axes[1,1].set_xlabel(r"$\cv[\tau]$")
axes[1,1].legend([r'$\E[\tau]={}$'.format(mu) for mu in mu_range])

##axes[0,1].legend([r"$\E[R_0]=1.82\times10^4$",
##                  r"$\E[R_0]=1.50\times10^4$",
##                  r"$\E[R_0]=7.16\times10^3$"])
##axes[1,0].legend([r"$\E[R_0]=6.69\times10^4$",
##                  r"$\E[R_0]=1.61\times10^5$",
##                  r"$\E[R_0]=2.48\times10^5$"])

plt.show()
