#!/usr/bin/env python
import Classes_Gillespie as Classy
import Functions_Gillespie as Gill
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle as pkl
import time as t
import datetime as dt
import multiprocessing as mp
safeProcessors = max(1, int(mp.cpu_count() * .5) - 1)


# Function Definitions
def run_pipeline(gillespie_parameters, processing_parameters, work_path, storage_path):
    [reaction_list, initial_state, system_size] = Initialize_Reactions(gillespie_parameters)
    [number_of_hits, thresh] = processing_parameters
    signal = Gill.gillespie_thresholding(reaction_list, number_of_hits, initial_state, threshold=thresh)
    archive_signal(signal, work_path + '{}transitions.pkl'.format(gillespie_parameters), storage_path)
    return signal

def archive_signal(signal, file_name, storage):
    os.system("touch " + file_name.replace(" ",r"\ "))
    with open(file_name, 'wb') as handle:
        pkl.dump(signal, handle, protocol=pkl.HIGHEST_PROTOCOL)
    os.system("gzip {} ;".format("'" + file_name + "'") +
              " mv {} {} &".format("'" + file_name + ".gz'", storage.replace(" ",r"\ ")))


def get_files(path, string="signal"):
    only_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    only_files = [f for f in only_files if string in f]
    return only_files


def Initialize_Reactions(parameters):
    [mu, cv] = parameters
    muA = muB = .3
    kb = kf = kon = 5
    kA = kB = koff = 1
    initial_vector = np.array([0,0,0,0,0,1,0], dtype=int)

    dilution0 = Classy.Reaction(np.array([-1,0,0,0,0,0,0], dtype=int), [0], 'mobius_propensity', [0, muA, 1, 0], 1, [0])
    dilution1 = Classy.Reaction(np.array([0,-1,0,0,0,0,0], dtype=int), [1], 'mobius_propensity', [0, muB, 1, 0], 1, [0])
    production0 = Classy.Reaction(np.array([1,0,0,0,0,0,0], dtype=int), [4], 'mobius_propensity', [0, kA, 1, 0], 'gamma_distribution', [mu,mu*cv])
    production1 = Classy.Reaction(np.array([0,1,0,0,0,0,0], dtype=int), [4], 'mobius_propensity', [0, kB, 1, 0], 'gamma_distribution', [mu,mu*cv])
    bound_production0 = Classy.Reaction(np.array([1,0,0,0,0,0,0], dtype=int), [5], 'mobius_propensity', [0, kA, 1, 0], 'gamma_distribution', [mu,mu*cv])
    bound_production1 = Classy.Reaction(np.array([0,1,0,0,0,0,0], dtype=int), [6], 'mobius_propensity', [0, kB, 1, 0], 'gamma_distribution', [mu,mu*cv])
    dimerization0 = Classy.Reaction(np.array([-2,0,1,0,0,0,0], dtype=int), [0], 'binomial_propensity', [2, kf], 1, [0])
    dimerization1 = Classy.Reaction(np.array([0,-2,0,1,0,0,0], dtype=int), [1], 'binomial_propensity', [2, kf], 1, [0])
    monimerization0 = Classy.Reaction(np.array([2,0,-1,0,0,0,0], dtype=int), [2], 'mobius_propensity', [0, kb, 1, 0], 1, [0])
    monimerization1 = Classy.Reaction(np.array([0,2,0,-1,0,0,0], dtype=int), [3], 'mobius_propensity', [0, kb, 1, 0], 1, [0])
    binding0 = Classy.Reaction(np.array([0,0,-1,0,-1,1,0], dtype=int), [2,4], 'product_propensity', [kon], 1, [0])
    binding1 = Classy.Reaction(np.array([0,0,0,-1,-1,0,1], dtype=int), [3,4], 'product_propensity', [kon], 1, [0])
    unbinding0 = Classy.Reaction(np.array([0,0,1,0,1,-1,0], dtype=int), [5], 'mobius_propensity', [0, koff, 1, 0], 1, [0])
    unbinding1 = Classy.Reaction(np.array([0,0,0,1,1,0,-1], dtype=int), [6], 'mobius_propensity', [0, koff, 1, 0], 1, [0])

    reaction_list = [dilution0, dilution1,
                     production0, production1,
                     bound_production0, bound_production1,
                     dimerization0, dimerization1,
                     monimerization0, monimerization1,
                     binding0, binding1,
                     unbinding0, unbinding1]
    return [reaction_list, initial_vector, 1]


if __name__ == '__main__':
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    os.system('gunzip ./propensity_table_6.pkl.gz')
    with open('./propensity_table_6.pkl', 'rb') as fh:
        prop_tab = pkl.load(fh)
    os.system('gzip ./propensity_table_6.pkl')
    Gill.set_propensity_table(prop_tab)

    with mp.Pool(safeProcessors) as pool2:
        mu_range = [15,20,25]
        cv_range = [thing.item() for thing in list(np.round(np.linspace(0,0.6,61),2))]
        number_of_hits = 21
        batches = 200
        thresh = 5
        paths_to_raw_data = ["2025-10-12/batch{}/".format(batch) for batch in range(batches)] ### edit to today's date and current batch
        paths_to_storage = ["data_storage/" + path for path in paths_to_raw_data]
        for batch in range(batches):
            Path(paths_to_raw_data[batch]).mkdir(parents=True, exist_ok=True)
            Path(paths_to_storage[batch]).mkdir(parents=True, exist_ok=True)

        parameter_sets = Gill.list_for_parallelization([mu_range, cv_range])
        pool_data = []
        for batch in range(batches):
            pool_data = pool_data + [{"parameters" : parameter_set, "batch" : batch} for parameter_set in parameter_sets]

        for batch in range(batches):
            existing_files = [f for f in get_files(paths_to_storage[batch], "transitions")
                              if os.path.getsize(os.path.join(paths_to_storage[batch], f)) > 1]
            for file in existing_files:
                index_extract = [(pool["batch"] == batch) for pool in pool_data]
                for index in [i for i,x in enumerate(index_extract) if x]:
                    if str(pool_data[index]["parameters"]) in file:
                        del pool_data[index]
                        break
        print(parameter_sets)
        try:
            start_time = t.time()
            pool2.starmap(run_pipeline, [(pool["parameters"], [number_of_hits, thresh],
                                          paths_to_raw_data[pool["batch"]], paths_to_storage[pool["batch"]]) for pool in pool_data])
            print(t.time()-start_time)
        finally:
            pool2.close()
            pool2.join()


