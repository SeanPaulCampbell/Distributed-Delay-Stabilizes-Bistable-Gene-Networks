import os
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import Functions_Gillespie as Gill
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}\DeclareMathOperator{\E}{\mathbb{E}}\DeclareMathOperator{\cv}{cv}')

mu_range = [4] 
cv_range = [thing.item() for thing in np.round(np.linspace(0,1.2,31),2)]
cvs = len(cv_range)
run_range = list(range(25))
state_range = [0,1]
states = len(state_range)
rate_range = [2**(-20)]
rates = len(rate_range)
system = "CR4Gamma_thresh"
path_to_storage = "2025-4-27/gamma_distributed/" ### edit to the correct path!
parameter_sets = Gill.list_for_parallelization([mu_range, cv_range, state_range, rate_range])
files = [[path_to_storage + 'batch{}/'.format(run) + '{}transitions.pkl.gz'.format(gillespie_parameters)
          for gillespie_parameters in parameter_sets]
         for run in run_range]
#files = [path_to_storage + '{}transitions.pkl.gz'.format(gillespie_parameters)
#         for gillespie_parameters in parameter_sets]

mean_residencies = np.zeros([rates, cvs]) 
stdev_residencies = np.zeros([rates, cvs]) 
cv_residencies = np.zeros([rates, cvs]) 
##mean_high_residencies = np.zeros([1,len(cv_range)]) 
##stdev_high_residencies = np.zeros([1,len(cv_range)]) 
##cv_high_residencies = np.zeros([1,len(cv_range)]) 
##mean_low_residencies = np.zeros([1,len(cv_range)]) 
##stdev_low_residencies = np.zeros([1,len(cv_range)]) 
##cv_low_residencies = np.zeros([1,len(cv_range)]) 

def unzip_data(file):
    os.system("gunzip "+file.replace(" ",r"\ "))
    with open(file[:-3], "rb") as read:
        data = pkl.load(read)
    os.system("gzip "+file[:-3].replace(" ",r"\ "))
    return data

residency_counts = [[0 for cv in cv_range] for rate in rate_range]
total_residencies = [[[] for cv in cv_range] for rate in rate_range]
for rate in rate_range:
    for cv in cv_range:
        cv_index = cv_range.index(cv)
        rate_index = rate_range.index(rate)
##    total_high = []
##    total_low = []
##    transitions = unzip_data(files[index])
##    residencies = np.diff([transition["time"] for transition in transitions[1:]])
##    if transitions[1]["high_low"] == 1:
##        high_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 0])
##        low_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 1])
##    else:
##        high_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 1])
##        low_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 0])
        
        for run in run_range:
            if os.path.exists(files[run][cv_index+cvs*(0+rate_index*states)]) and os.path.exists(files[run][cv_index+cvs*(1+rate_index*states)]):
                transitions = unzip_data(files[run][cv_index+cvs*(0+rate_index*states)]) # have to change this for when we do mu and cv
                residencies = list(np.diff([transition["time"] for transition in transitions]))
                if transitions[0]["intermediate"]:
                    residencies = [residencies[index] for index in range(len(residencies)) if index % 2 == 1]
                else:
                    residencies = [residencies[index] for index in range(len(residencies)) if index % 2 == 0]
                total_residencies[rate_index][cv_index] += residencies
##                plt.plot([tran["time"] for tran in transitions],[tran["state"][0] for tran in transitions])
##                plt.plot([tran["time"] for tran in transitions],[tran["state"][1] for tran in transitions])
##                plt.scatter([tran["time"] for tran in transitions],[tran["state"][0] for tran in transitions])
##                plt.scatter([tran["time"] for tran in transitions],[tran["state"][1] for tran in transitions])
##                plt.show()
                transitions = unzip_data(files[run][cv_index+cvs*(1+rate_index*states)]) # have to change this for when we do mu and cv
                residencies = list(np.diff([transition["time"] for transition in transitions]))
                if transitions[0]["intermediate"]:
                    residencies = [residencies[index] for index in range(len(residencies)) if index % 2 == 1]
                else:
                    residencies = [residencies[index] for index in range(len(residencies)) if index % 2 == 0]
                total_residencies[rate_index][cv_index] += residencies

##            if transitions[1]["high_low"] == 1:
##                high_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 0])
##                low_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 1])
##            else:
##                high_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 1])
##                low_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 0])
##            total_high += list(high_residencies)
##            total_low += list(low_residencies) 
        residency_counts[rate_index][cv_index] = len(total_residencies[rate_index][cv_index])
    
##    mean_high_residencies[0,index] = np.mean(total_high)
##    stdev_high_residencies[0,index] = np.std(total_high)
##    
##    mean_low_residencies[0,index] = np.mean(total_low)
##    stdev_low_residencies[0,index] = np.std(total_low)

        mean_residencies[rate_index,cv_index] = np.mean(total_residencies[rate_index][cv_index])
        stdev_residencies[rate_index,cv_index] = np.std(total_residencies[rate_index][cv_index])
sterr_residencies = np.divide(stdev_residencies, np.sqrt(np.array(residency_counts, dtype=int)))
cv_residencies = np.divide(stdev_residencies, mean_residencies)

np.save(path_to_storage+system+"residency_counts.npy", np.array(residency_counts,dtype=int))
np.save(path_to_storage+system+"mean_residencies.npy", mean_residencies)
np.save(path_to_storage+system+"cv_residencies.npy", cv_residencies)

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

fig, axes = plt.subplots(2)
sns.histplot(pd.concat([pd.DataFrame([{r"$\cv[\tau]$" : cv_range[index],
                                       "time to hit separatrix" : total_residencies[0][index][jndex]}
                                      for jndex in range(len(total_residencies[0][index]))])
                        for index in range(cvs)]),
             x="time to hit separatrix", ax = axes[0], hue=r"$\cv[\tau]$", palette="icefire",
             common_norm=False, log_scale=True, kde=True, stat="density", element="step", alpha=0.2)
fig.suptitle(r"histograms of transition attempt times for gamma distributed delay vs $\cv[\tau]$")
sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1.3))
axes[1].errorbar(cv_range, mean_residencies[0], yerr=sterr_residencies[0], fmt='o',
                 capsize=5, ms=2, ecolor='cornflowerblue', mfc='royalblue', mec='royalblue')
axes[1].set_title(r"Mean Transition Attempt Time")
axes[1].set_xlabel(r"$\cv[\tau]$")
##axes[1].set_yscale('log')
axes[1].set_ylabel(r"$\E[\text{attempt time}]$")
plt.show()

####figure, axes = plt.subplots(3,2)
##figure, axes = plt.subplots(2)
##figure.suptitle("Switching Corepressive")
##
##sns.heatmap(mean_residencies, ax=axes[0])
##axes[0].set_title("Residency Mean")
##axes[0].set_xlabel("Bernoulli CV")
##axes[0].set_xticklabels(cv_range)
##axes[0].set_ylabel("Log_2(switching rate)")
##axes[0].set_yticklabels(log_rate_range)
##sns.heatmap(cv_residencies, ax=axes[1])
##axes[1].set_title("Residency CV")
##axes[1].set_xlabel("Bernoulli CV")
##axes[1].set_xticklabels(cv_range)
##axes[1].set_ylabel("Log_2(switching rate)")
##axes[1].set_yticklabels(log_rate_range)
##plt.tight_layout()
##
##
####axes[0,0].plot(cv_range, mean_residencies[0,:]) 
####axes[0,1].plot(cv_range, cv_residencies[0,:]) 
####axes[1,0].plot(cv_range, mean_high_residencies[0,:]) 
####axes[1,1].plot(cv_range, cv_high_residencies[0,:]) 
####axes[2,0].plot(cv_range, mean_low_residencies[0,:]) 
####axes[2,1].plot(cv_range, cv_low_residencies[0,:]) 
####axes[0].plot(log_rate_range, mean_residencies[:,0])
####axes[1].plot(log_rate_range, cv_residencies[:,0])
####axes[0].plot(log_rate_range, mean_residencies[:,1])
####axes[1].plot(log_rate_range, cv_residencies[:,1])
####axes[0].plot(log_rate_range, 75148.25759057*np.ones(rates))
####axes[0].plot(log_rate_range, 121114.67248298273*np.ones(rates))
####axes[0,0].set_title('Mean Residency Time')
####axes[0,1].set_title('Residency Time CV')
####axes[0].set_title('Mean Residency Times')
####axes[1].set_title('Residency Time CV')
####axes[0].set_ylabel('Mean Residency Time')
####axes[1].set_ylabel('cv[R]')
####axes[1].set_xlabel('Log_4(Switching Rate)')
####axes[1].set_ylim(0,2)
####axes[0].legend(['cv=.01','cv=.6','cv=0','distributed cv=.6'])
####axes[1,0].set_title('High State Mean Residency Time')
####axes[1,1].set_title('High State Residency Time CV')
####axes[2,0].set_title('Low State Mean Residency Time')
####axes[2,1].set_title('Low State Residency Time CV')
##plt.show()
##
