import os
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import Functions_Gillespie as Gill

mu_range = [2,3,4] 
cv_range = [thing.item() for thing in np.round(np.linspace(0,1.2,31),2)]
run_range = list(range(50))
systems = ["CR{}".format(mu) for mu in mu_range]
path_to_storage = "2025-2-4/" ### edit to the correct path!
parameter_sets = Gill.list_for_parallelization([mu_range, cv_range])
files = [[path_to_storage + 'batch{}/'.format(run) + '{}transitions.pkl.gz'.format(gillespie_parameters)
          for gillespie_parameters in parameter_sets]
         for run in run_range]
#files = [path_to_storage + '{}transitions.pkl.gz'.format(gillespie_parameters)
#         for gillespie_parameters in parameter_sets]
mus = len(mu_range)
cvs = len(cv_range)
mean_residencies = np.zeros([mus,cvs]) 
stdev_residencies = np.zeros([mus,cvs]) 
cv_residencies = np.zeros([mus,cvs])
residency_counts = np.zeros([mus,cvs])
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

for mu in mu_range:
    mu_index = mu_range.index(mu)
    for cv in cv_range:
        cv_index = cv_range.index(cv)
        index = parameter_sets.index([mu,cv])
        total_residencies = []
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
            try:
                transitions = unzip_data(files[run][index]) # have to change this for when we do mu and cv
            #print(transitions[-1]["time"])
                residencies = np.diff([transition["time"] for transition in transitions[1:]])
                total_residencies += list(residencies)
##            if transitions[1]["high_low"] == 1:
##                high_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 0])
##                low_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 1])
##            else:
##                high_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 1])
##                low_residencies = np.array([residencies[index0] for index0 in range(len(residencies)) if index0 % 2 == 0])
##            total_high += list(high_residencies)
##            total_low += list(low_residencies) 
            except:
                continue
        residency_counts[mu_index,cv_index] = len(total_residencies)

##    mean_high_residencies[0,index] = np.mean(total_high)
##    stdev_high_residencies[0,index] = np.std(total_high)
##    
##    mean_low_residencies[0,index] = np.mean(total_low)
##    stdev_low_residencies[0,index] = np.std(total_low)

        mean_residencies[mu_index,cv_index] = np.mean(total_residencies)
        stdev_residencies[mu_index,cv_index] = np.std(total_residencies)

print(residency_counts)

print(stdev_residencies)
print(mean_residencies)
cv_residencies = np.divide(stdev_residencies, mean_residencies)
print(cv_residencies)
##cv_high_residencies = np.divide(stdev_high_residencies, mean_high_residencies)
##cv_low_residencies = np.divide(stdev_low_residencies, mean_low_residencies)
for system in systems:
    index = systems.index(system)
    np.save(path_to_storage+system+"mean_residencies.npy", mean_residencies[index,:])
##np.save(path_to_storage+"mean_low_residencies.npy", mean_low_residencies)
##np.save(path_to_storage+"mean_high_residencies.npy", mean_high_residencies)
    np.save(path_to_storage+system+"cv_residencies.npy", cv_residencies[index,:])
    np.save(path_to_storage+system+"residency_count.npy", residency_counts[index,:])
##np.save(path_to_storage+"cv_low_residencies.npy", cv_low_residencies)
##np.save(path_to_storage+"cv_high_residencies.npy", cv_high_residencies)
##mean_normalizer = mean_residencies[0,0]
##mean_residencies[0,:] = mean_residencies[0,:] / mean_residencies[0,0]
##mean_low_residencies[0,:] = mean_low_residencies[0,:] / mean_low_residencies[0,0]
##mean_high_residencies[0,:] = mean_high_residencies[0,:] / mean_high_residencies[0,0]

##figure, axes = plt.subplots(3,2)
##figure, axes = plt.subplots(2)
##figure.suptitle("Corepressive Toggle mean 3 Bernoulli Distributed")

##axes[0,0].plot(cv_range, mean_residencies[0,:]) 
##axes[0,1].plot(cv_range, cv_residencies[0,:]) 
##axes[1,0].plot(cv_range, mean_high_residencies[0,:]) 
##axes[1,1].plot(cv_range, cv_high_residencies[0,:]) 
##axes[2,0].plot(cv_range, mean_low_residencies[0,:]) 
##axes[2,1].plot(cv_range, cv_low_residencies[0,:]) 
##axes[0].plot(cv_range, mean_residencies[0,:])
##axes[1].plot(cv_range, cv_residencies[0,:])

##axes[0,0].set_title('Mean Residency Time')
##axes[0,1].set_title('Residency Time CV')
##axes[0].set_title('Mean Residency Time Normalized by R0={}'.format(round(mean_normalizer)))
##axes[1].set_title('Residency Time CV')
##axes[0].set_ylabel('R/R0')
##axes[1].set_ylabel('cv[R]')
##axes[1].set_xlabel('Delay CV')
##axes[1].set_ylim(0,2)
##axes[1,0].set_title('High State Mean Residency Time')
##axes[1,1].set_title('High State Residency Time CV')
##axes[2,0].set_title('Low State Mean Residency Time')
##axes[2,1].set_title('Low State Residency Time CV')
##plt.show()
