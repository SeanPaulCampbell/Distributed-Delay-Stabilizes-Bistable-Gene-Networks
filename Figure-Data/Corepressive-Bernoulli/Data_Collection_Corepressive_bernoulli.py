import os
import math
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import Functions_Gillespie as Gill

mu_range = [2,3,4]
mus = len(mu_range)
cv_range = [thing.item() for thing in np.round(np.linspace(0,1.2,31),2)]
cvs = len(cv_range)
run_range = list(range(100))
state_range = [0,1]
states = len(state_range)
##log_rate_range = np.linspace(0,8,3)
rate_range = [1] # [thing.item() for thing in 2**log_rate_range]
##log_rate_range = [thing.item() for thing in log_rate_range]
##rates = len(rate_range)
system = "CR_bernoulli"
path_to_storage = "2025-8-21/" ### edit to the correct path!
parameter_sets = Gill.list_for_parallelization([mu_range, cv_range, state_range, rate_range])
files = [[path_to_storage + 'batch{}/'.format(run) + '{}transitions.pkl.gz'.format(gillespie_parameters)
          for gillespie_parameters in parameter_sets]
         for run in run_range]
#files = [path_to_storage + '{}transitions.pkl.gz'.format(gillespie_parameters)
#         for gillespie_parameters in parameter_sets]

mean_residencies = np.zeros([mus, cvs]) 
stdev_residencies = np.zeros([mus, cvs]) 
cv_residencies = np.zeros([mus, cvs])
geo_mean = np.zeros([mus, cvs])
geo_std = np.zeros([mus, cvs])
geo_cv = np.zeros([mus, cvs])
att_mean = np.zeros([mus, cvs])
att_std = np.zeros([mus, cvs])
att_cv = np.zeros([mus, cvs])

def unzip_data(file):
    os.system("gunzip "+file.replace(" ",r"\ "))
    with open(file[:-3], "rb") as read:
        data = pkl.load(read)
    os.system("gzip "+file[:-3].replace(" ",r"\ "))
    return data

residency_counts = [[0 for cv in cv_range] for mu in mu_range]
attempt_counts = [[0 for cv in cv_range] for mu in mu_range]
for mu in mu_range:
    for cv in cv_range:
        cv_index = cv_range.index(cv)
        mu_index = mu_range.index(mu)
        total_residencies = []
        total_geometrics = []
        total_attempts = []
        for run in run_range:
##            if os.path.exists(files[run][cv_index+cvs*(0+rate_index*states)]) and os.path.exists(files[run][cv_index+cvs*(1+rate_index*states)]):
            for state in range(2):
                if os.path.exists(files[run][mu_index + mus*(cv_index + cvs*state)]):
                    data = unzip_data(files[run][mu_index + mus*(cv_index + cvs*state)]) # have to change this for when we do mu and cv
                    residencies = np.diff([datum["time"] for datum in data[1:] if datum["transition"]])
                    geometric_data = [datum["hits"] for datum in data[1:] if datum["transition"]]
                    attempts_data = list(np.diff([datum["time"] for datum in data[1:]]))
                    if data[1]["intermediate"]:
                        attempts_data = [attempts_data[index] for index in range(len(attempts_data)) if index % 2 == 1]
                    else:
                        attempts_data = [attempts_data[index] for index in range(len(attempts_data)) if index % 2 == 0]
                    total_residencies += list(residencies)
                    total_geometrics += geometric_data
                    total_attempts += attempts_data
        if total_attempts:
            residency_counts[mu_index][cv_index] = len(total_residencies)
            attempt_counts[mu_index][cv_index] = len(total_attempts)
            mean_residencies[mu_index,cv_index] = np.mean(total_residencies)
            stdev_residencies[mu_index,cv_index] = np.std(total_residencies)
            geo_mean[mu_index,cv_index] = np.mean(total_geometrics)
            geo_std[mu_index,cv_index] = np.std(total_geometrics)
            att_mean[mu_index,cv_index] = np.mean(total_attempts)
            att_std[mu_index,cv_index] = np.std(total_attempts)
            

cv_residencies = np.divide(stdev_residencies, mean_residencies)
geo_cv = np.divide(geo_std, geo_mean)
att_cv = np.divide(att_std, att_mean)
print(mean_residencies)
print(geo_mean)
print(att_mean)
print(residency_counts)
np.save(path_to_storage+system+"_residency_counts.npy", np.array(residency_counts,dtype=int))
np.save(path_to_storage+system+"_mean_residencies.npy", mean_residencies)
np.save(path_to_storage+system+"_cv_residencies.npy", cv_residencies)
np.save(path_to_storage+system+"_attempt_counts.npy", np.array(attempt_counts,dtype=int))
np.save(path_to_storage+system+"_thresh_means.npy", att_mean)
np.save(path_to_storage+system+"_thresh_cvs.npy", att_cv)
np.save(path_to_storage+system+"_geometric_means.npy", geo_mean)
np.save(path_to_storage+system+"_geometric_cvs.npy", geo_cv)

